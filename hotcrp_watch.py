#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import difflib
import getpass
import hashlib
import http.cookiejar
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Comment


SCORE_FIELD_PATTERNS = [
    "score",
    "overall",
    "overall merit",
    "merit",
    "recommendation",
    "confidence",
    "expertise",
    "reviewer expertise",
]
DEFAULT_PUSHPLUS_URL = "https://www.pushplus.plus/send"


@dataclasses.dataclass(frozen=True)
class Target:
    paper_id: str
    url: str


@dataclasses.dataclass(frozen=True)
class PageSnapshot:
    html: str
    text: str
    parsed: dict


@dataclasses.dataclass(frozen=True)
class SaveResult:
    changed: bool
    snapshot_html: Path
    snapshot_text: Path
    snapshot_json: Path
    diff_path: Path


class SafeSession(requests.Session):
    WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

    def __init__(self, site: str):
        super().__init__()
        self.site = normalize_site(site)

    def check_write_allowed(self, method: str, url: str) -> None:
        method = method.upper()
        if method not in self.WRITE_METHODS:
            return
        absolute = urljoin(self.site + "/", url)
        parsed = urlparse(absolute)
        site = urlparse(self.site)
        if parsed.scheme == site.scheme and parsed.netloc == site.netloc and parsed.path == "/signin":
            return
        raise RuntimeError(f"Refusing {method} to non-login URL: {absolute}")

    def request(self, method, url, *args, **kwargs):
        self.check_write_allowed(str(method), str(url))
        return super().request(method, url, *args, **kwargs)


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def normalize_site(site: str) -> str:
    return site.rstrip("/")


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def paper_id_from_url(url: str) -> str:
    match = re.search(r"/paper/([^/?#]+)", url)
    if match:
        return match.group(1)
    parsed = urlparse(url)
    return parsed.path.strip("/").replace("/", "-") or hashlib.sha1(url.encode()).hexdigest()[:12]


def parse_targets(args: argparse.Namespace) -> list[Target]:
    seen: dict[str, Target] = {}
    site = normalize_site(args.site)

    for paper_id in split_csv(args.paper):
        target = Target(paper_id=paper_id, url=f"{site}/paper/{paper_id}")
        seen[target.paper_id] = target

    for raw_url in split_csv(args.url):
        url = raw_url if raw_url.startswith(("http://", "https://")) else urljoin(site + "/", raw_url)
        paper_id = paper_id_from_url(url)
        seen.setdefault(paper_id, Target(paper_id=paper_id, url=url))

    if not seen:
        raise SystemExit("Provide at least one --paper id list or --url list.")
    return list(seen.values())


def node_hidden(tag) -> bool:
    if not getattr(tag, "attrs", None):
        return False
    if tag.has_attr("hidden") or tag.get("aria-hidden") == "true":
        return True
    style = tag.get("style", "")
    if isinstance(style, str) and re.search(r"display\s*:\s*none|visibility\s*:\s*hidden", style, re.I):
        return True
    if tag.name == "input" and str(tag.get("type", "")).lower() == "hidden":
        return True
    return False


def remove_invisible_nodes(soup: BeautifulSoup) -> None:
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for tag in list(soup.find_all(True)):
        if node_hidden(tag):
            tag.decompose()


def extract_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    remove_invisible_nodes(soup)
    lines = []
    for line in soup.get_text("\n").splitlines():
        cleaned = " ".join(line.split())
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def has_score_context(text: str) -> bool:
    lowered = text.lower().replace("_", " ")
    return any(pattern in lowered for pattern in SCORE_FIELD_PATTERNS)


def find_visible_score_fields(text: str) -> list[dict]:
    fields = []
    for line in text.splitlines():
        if has_score_context(line):
            match = re.search(r"([A-Za-z][A-Za-z _-]{1,40})\s*[:=]\s*([^;,\n]{1,80})", line)
            if match:
                fields.append({"field": match.group(1).strip(), "value_snippet": match.group(2).strip()})
    return fields


def short_context(text: str, max_len: int = 160) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def artifact(field: str, value: str, location_type: str, context: str) -> dict:
    return {
        "field": field,
        "value_snippet": short_context(value, 120),
        "location_type": location_type,
        "selector_or_context": short_context(context, 160),
    }


def scan_hidden_dom(soup: BeautifulSoup) -> list[dict]:
    found = []
    for tag in soup.find_all(True):
        if not node_hidden(tag):
            continue
        name = tag.get("name") or tag.get("id") or "hidden"
        value = tag.get("value") or tag.get_text(" ", strip=True)
        context = f"<{tag.name} {short_context(tag.attrs)}>"
        if value and has_score_context(f"{name} {value} {context}"):
            found.append(artifact(str(name), str(value), "hidden_dom", context))
    return found


def scan_dom_attributes(soup: BeautifulSoup) -> list[dict]:
    found = []
    for tag in soup.find_all(True):
        for key, value in tag.attrs.items():
            if not key.startswith("data-"):
                continue
            joined = " ".join(value) if isinstance(value, list) else str(value)
            if has_score_context(f"{key} {joined}"):
                found.append(artifact(key, joined, "dom_attribute", f"<{tag.name} {key}=...>"))
    return found


def scan_inline_jsonish(soup: BeautifulSoup) -> list[dict]:
    found = []
    pattern = re.compile(
        r'["\']?([A-Za-z_ -]*(?:score|overall|merit|recommendation|confidence|expertise)[A-Za-z_ -]*)["\']?\s*[:=]\s*["\']?([^,"\';}\]]{1,80})',
        re.I,
    )
    for script in soup.find_all("script"):
        body = script.string or script.get_text(" ", strip=True)
        for match in pattern.finditer(body):
            found.append(artifact(match.group(1).strip(), match.group(2).strip(), "inline_json", match.group(0)))
    return found


def scan_html_comments(soup: BeautifulSoup) -> list[dict]:
    found = []
    pattern = re.compile(
        r"([A-Za-z_ -]*(?:score|overall|merit|recommendation|confidence|expertise)[A-Za-z_ -]*)\s*[:=]\s*([^;\n,]{1,80})",
        re.I,
    )
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        for match in pattern.finditer(str(comment)):
            found.append(artifact(match.group(1).strip(), match.group(2).strip(), "html_comment", str(comment)))
    return found


def scan_delivered_artifacts(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    artifacts = []
    artifacts.extend(scan_hidden_dom(soup))
    artifacts.extend(scan_dom_attributes(soup))
    artifacts.extend(scan_inline_jsonish(soup))
    artifacts.extend(scan_html_comments(soup))
    return artifacts


def extract_visible_blocks(text: str) -> list[str]:
    keywords = ("review", "comment", "response", "author response")
    blocks = []
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if any(keyword in line.lower() for keyword in keywords):
            block = [line]
            for follower in lines[idx + 1 : idx + 5]:
                if any(keyword in follower.lower() for keyword in keywords):
                    break
                block.append(follower)
            blocks.append(short_context("\n".join(block), 800))
    return blocks or [short_context(text, 1200)] if text else []


def stable_fingerprint(text: str) -> str:
    normalized = re.sub(r"(?i)(token|nonce|csrf|post|salt|session)[-_a-z0-9]*\s*[:=]\s*[\"']?[-_.a-z0-9]{8,}", r"\1=<volatile>", text)
    normalized = re.sub(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:?\d{2})?\b", "<timestamp>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()


def parse_page(paper_id: str, url: str, html: str) -> dict:
    text = extract_visible_text(html)
    return {
        "paper_id": paper_id,
        "page_url": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "login_ok": not looks_like_login_page(html),
        "visible_blocks": extract_visible_blocks(text),
        "visible_score_fields": find_visible_score_fields(text),
        "delivered_artifacts": scan_delivered_artifacts(html),
        "raw_text_fingerprint": stable_fingerprint(text),
        "change_summary": [],
    }


def looks_like_login_page(html: str) -> bool:
    lowered = html.lower()
    return ("type=\"password\"" in lowered or "name=\"password\"" in lowered) and ("signin" in lowered or "sign in" in lowered or "login" in lowered)


def extract_page_messages(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    messages = []
    for tag in soup.select(".msg, .msg-error, .msg-warning, .feedback, .is-error"):
        text = short_context(tag.get_text(" ", strip=True), 300)
        if text and text not in messages:
            messages.append(text)
    return messages


def save_login_failure_diagnostic(out_dir: Path, html: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "login-failure.html"
    text_path = out_dir / "login-failure.txt"
    write_text_secure(html_path, html)
    messages = extract_page_messages(html)
    visible_text = extract_visible_text(html)
    if messages:
        text = "\n".join(messages) + "\n"
    else:
        text = short_context(visible_text, 4000) + "\n"
    write_text_secure(text_path, text)
    return html_path


def paper_dir(out_dir: Path, target: Target) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", target.paper_id)
    return out_dir / f"paper-{safe_id}"


def write_text_secure(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    write_text_secure(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def save_baseline(out_dir: Path, target: Target, snapshot: PageSnapshot) -> None:
    directory = paper_dir(out_dir, target)
    directory.mkdir(parents=True, exist_ok=True)
    write_text_secure(directory / "baseline.html", snapshot.html)
    write_text_secure(directory / "baseline.txt", snapshot.text)
    write_json(directory / "baseline.json", snapshot.parsed)
    write_latest(directory, snapshot)


def write_latest(directory: Path, snapshot: PageSnapshot) -> None:
    write_text_secure(directory / "latest.html", snapshot.html)
    write_text_secure(directory / "latest.txt", snapshot.text)
    write_json(directory / "latest.json", snapshot.parsed)


def load_latest(out_dir: Path, target: Target) -> PageSnapshot | None:
    directory = paper_dir(out_dir, target)
    html_path = directory / "latest.html"
    text_path = directory / "latest.txt"
    json_path = directory / "latest.json"
    if not (html_path.exists() and text_path.exists() and json_path.exists()):
        return None
    return PageSnapshot(
        html=html_path.read_text(encoding="utf-8", errors="replace"),
        text=text_path.read_text(encoding="utf-8", errors="replace"),
        parsed=json.loads(json_path.read_text(encoding="utf-8")),
    )


def snapshots_changed(old: PageSnapshot, new: PageSnapshot) -> bool:
    old_artifacts = json.dumps(old.parsed.get("delivered_artifacts", []), sort_keys=True)
    new_artifacts = json.dumps(new.parsed.get("delivered_artifacts", []), sort_keys=True)
    old_blocks = json.dumps(old.parsed.get("visible_blocks", []), sort_keys=True)
    new_blocks = json.dumps(new.parsed.get("visible_blocks", []), sort_keys=True)
    old_scores = json.dumps(old.parsed.get("visible_score_fields", []), sort_keys=True)
    new_scores = json.dumps(new.parsed.get("visible_score_fields", []), sort_keys=True)
    return (
        old.parsed.get("raw_text_fingerprint") != new.parsed.get("raw_text_fingerprint")
        or old_artifacts != new_artifacts
        or old_blocks != new_blocks
        or old_scores != new_scores
    )


def save_change_snapshot(out_dir: Path, target: Target, old: PageSnapshot, new: PageSnapshot, stamp: str | None = None) -> SaveResult:
    stamp = stamp or now_stamp()
    directory = paper_dir(out_dir, target)
    directory.mkdir(parents=True, exist_ok=True)
    changed = snapshots_changed(old, new)
    html_path = directory / f"snapshot-{stamp}.html"
    text_path = directory / f"snapshot-{stamp}.txt"
    json_path = directory / f"snapshot-{stamp}.json"
    diff_path = directory / f"diff-{stamp}.txt"

    if changed:
        write_text_secure(html_path, new.html)
        write_text_secure(text_path, new.text)
        write_json(json_path, new.parsed)
        diff = difflib.unified_diff(
            old.text.splitlines(),
            new.text.splitlines(),
            fromfile="previous",
            tofile=f"snapshot-{stamp}",
            lineterm="",
        )
        write_text_secure(diff_path, "\n".join(diff) + "\n")
    write_latest(directory, new)
    return SaveResult(changed, html_path, text_path, json_path, diff_path)


def build_pushplus_payload(token: str, target: Target, result: SaveResult, topic: str | None = None) -> dict:
    content = "\n".join(
        [
            f"## HotCRP paper {target.paper_id} changed",
            "",
            f"- URL: {target.url}",
            f"- Diff: `{result.diff_path}`",
            f"- HTML snapshot: `{result.snapshot_html}`",
            f"- Text snapshot: `{result.snapshot_text}`",
            f"- JSON snapshot: `{result.snapshot_json}`",
        ]
    )
    payload = {
        "token": token,
        "title": f"HotCRP paper {target.paper_id} changed",
        "content": content,
        "template": "markdown",
    }
    if topic:
        payload["topic"] = topic
    return payload


def send_pushplus_notification(
    token: str,
    target: Target,
    result: SaveResult,
    topic: str | None = None,
    url: str = DEFAULT_PUSHPLUS_URL,
    session: requests.Session | None = None,
) -> dict:
    http = session or requests.Session()
    response = http.post(url, json=build_pushplus_payload(token, target, result, topic), timeout=20)
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError:
        payload = {"raw": response.text}
    code = payload.get("code")
    if code not in (None, 200):
        raise RuntimeError(f"PushPlus returned code={code}: {payload}")
    return payload


def pushplus_token_from_args(args: argparse.Namespace) -> str:
    return getattr(args, "pushplus_token", "") or os.environ.get("PUSHPLUS_TOKEN", "")


def pushplus_topic_from_args(args: argparse.Namespace) -> str:
    return getattr(args, "pushplus_topic", "") or os.environ.get("PUSHPLUS_TOPIC", "")


def maybe_send_pushplus(args: argparse.Namespace, target: Target, result: SaveResult) -> None:
    token = pushplus_token_from_args(args)
    if not token:
        return
    try:
        send_pushplus_notification(
            token,
            target,
            result,
            topic=pushplus_topic_from_args(args) or None,
            url=getattr(args, "pushplus_url", DEFAULT_PUSHPLUS_URL),
        )
        print(f"  pushplus=sent")
    except Exception as exc:
        print(f"  pushplus=ERROR: {exc}", file=sys.stderr)


def cookie_path(out_dir: Path) -> Path:
    return out_dir / "session.cookies"


def load_cookies(session: requests.Session, out_dir: Path) -> None:
    path = cookie_path(out_dir)
    jar = http.cookiejar.MozillaCookieJar(str(path))
    if path.exists():
        jar.load(ignore_discard=True, ignore_expires=True)
    session.cookies = jar


def save_cookies(session: requests.Session, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jar = session.cookies
    if hasattr(jar, "save"):
        jar.save(ignore_discard=True, ignore_expires=True)
    os.chmod(cookie_path(out_dir), 0o600)


def find_login_form(html: str):
    soup = BeautifulSoup(html, "lxml")
    password_input = soup.find("input", {"type": "password"})
    if password_input is None:
        return None, None, None
    form = password_input.find_parent("form")
    email_input = None
    if form:
        email_input = form.find("input", {"type": "email"}) or form.find("input", {"name": re.compile(r"(email|u|user)", re.I)})
    return form, email_input, password_input


def form_payload(form, email_input, password_input, email: str, password: str) -> dict:
    payload = {}
    if form:
        for inp in form.find_all("input"):
            name = inp.get("name")
            if not name:
                continue
            payload[name] = inp.get("value", "")
    payload[email_input.get("name") if email_input and email_input.get("name") else "email"] = email
    payload[password_input.get("name") if password_input.get("name") else "password"] = password
    return payload


def fetch_session_post_value(session: requests.Session, site: str) -> str:
    response = session.get(
        normalize_site(site) + "/api/session",
        headers={"Sec-Fetch-Site": "same-origin"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    post_value = payload.get("sessioninfo", {}).get("postvalue")
    if not post_value:
        raise RuntimeError("HotCRP /api/session did not return a postvalue.")
    return str(post_value)


def ensure_login(session: requests.Session, site: str, email: str, out_dir: Path, reset: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if reset and cookie_path(out_dir).exists():
        cookie_path(out_dir).unlink()
    load_cookies(session, out_dir)

    home = session.get(normalize_site(site) + "/", timeout=30)
    if not looks_like_login_page(home.text):
        save_cookies(session, out_dir)
        return

    form, email_input, password_input = find_login_form(home.text)
    if password_input is None:
        raise RuntimeError("Could not find a HotCRP password login form.")
    action = form.get("action") if form else ""
    login_url = urljoin(home.url, action or home.url)
    password = getpass.getpass(f"HotCRP password for {email}: ")
    payload = form_payload(form, email_input, password_input, email, password)
    payload["post"] = fetch_session_post_value(session, site)
    response = session.post(
        login_url,
        data=payload,
        headers={
            "Origin": normalize_site(site),
            "Referer": normalize_site(site) + "/",
        },
        timeout=30,
    )
    if looks_like_login_page(response.text):
        diagnostic_path = save_login_failure_diagnostic(out_dir, response.text)
        messages = extract_page_messages(response.text)
        detail = "; ".join(messages) if messages else "no HotCRP error message found"
        raise RuntimeError(
            "Login still appears to be on the password page. "
            f"HotCRP says: {detail}. Saved diagnostic page to {diagnostic_path}."
        )
    save_cookies(session, out_dir)


def fetch_snapshot(session: requests.Session, target: Target) -> PageSnapshot:
    response = session.get(target.url, timeout=30)
    response.raise_for_status()
    html = response.text
    text = extract_visible_text(html)
    parsed = parse_page(target.paper_id, target.url, html)
    return PageSnapshot(html=html, text=text, parsed=parsed)


def inspect_targets(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    session = SafeSession(args.site)
    ensure_login(session, args.site, args.email, out_dir, args.reset_session)
    targets = parse_targets(args)
    summary = {"fetched_at": datetime.now(timezone.utc).isoformat(), "papers": []}

    for target in targets:
        try:
            snapshot = fetch_snapshot(session, target)
            previous = load_latest(out_dir, target)
            if previous is None:
                save_baseline(out_dir, target, snapshot)
                changed = True
                diff_path = None
            else:
                result = save_change_snapshot(out_dir, target, previous, snapshot)
                changed = result.changed
                diff_path = str(result.diff_path) if result.changed else None
                if result.changed:
                    maybe_send_pushplus(args, target, result)
            summary["papers"].append({"paper_id": target.paper_id, "url": target.url, "changed": changed, "diff": diff_path})
            print_paper_report(target, snapshot, changed, diff_path)
        except Exception as exc:
            summary["papers"].append({"paper_id": target.paper_id, "url": target.url, "error": str(exc)})
            print(f"[paper {target.paper_id}] ERROR: {exc}", file=sys.stderr)
    write_json(out_dir / "latest-summary.json", summary)
    return 0


def print_paper_report(target: Target, snapshot: PageSnapshot, changed: bool, diff_path: str | None) -> None:
    parsed = snapshot.parsed
    print(f"[paper {target.paper_id}] changed={changed} login_ok={parsed.get('login_ok')}")
    print(f"  visible_blocks={len(parsed.get('visible_blocks', []))}")
    print(f"  visible_score_fields={len(parsed.get('visible_score_fields', []))}")
    print(f"  delivered_artifacts={len(parsed.get('delivered_artifacts', []))}")
    if diff_path:
        print(f"  diff={diff_path}")
    for item in parsed.get("visible_score_fields", [])[:5]:
        print(f"  visible: {item['field']}: {item['value_snippet']}")
    for item in parsed.get("delivered_artifacts", [])[:8]:
        print(f"  artifact[{item['location_type']}]: {item['field']}: {item['value_snippet']}")


def consume_reset_session(args: argparse.Namespace) -> bool:
    reset = bool(getattr(args, "reset_session", False))
    if reset:
        args.reset_session = False
    return reset


def watch_targets(args: argparse.Namespace) -> int:
    while True:
        reset_session = consume_reset_session(args)
        args.reset_session = reset_session
        inspect_targets(args)
        args.reset_session = False
        if getattr(args, "once", False):
            return 0
        time.sleep(args.interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and watch HotCRP paper pages.")
    subparsers = parser.add_subparsers(dest="command")

    def add_common(sub):
        sub.add_argument("--site", default="https://iccad2026.hotcrp.com")
        sub.add_argument("--email", required=True)
        sub.add_argument("--paper", default="")
        sub.add_argument("--url", default="")
        sub.add_argument("--out", default=str(Path(__file__).resolve().parent / "state"))
        sub.add_argument("--reset-session", action="store_true")
        sub.add_argument("--pushplus-token", default="", help="PushPlus token; defaults to PUSHPLUS_TOKEN.")
        sub.add_argument("--pushplus-topic", default="", help="Optional PushPlus topic; defaults to PUSHPLUS_TOPIC.")
        sub.add_argument("--pushplus-url", default=DEFAULT_PUSHPLUS_URL)

    inspect_parser = subparsers.add_parser("inspect", help="Fetch, parse, and save snapshots once.")
    add_common(inspect_parser)
    inspect_parser.set_defaults(func=inspect_targets)

    watch_parser = subparsers.add_parser("watch", help="Poll pages and save snapshots when they change.")
    add_common(watch_parser)
    watch_parser.add_argument("--interval", type=int, default=300)
    watch_parser.add_argument("--once", action="store_true")
    watch_parser.set_defaults(func=watch_targets)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
