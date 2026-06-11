import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "hotcrp_watch.py"
SPEC = importlib.util.spec_from_file_location("hotcrp_watch", MODULE_PATH)
hotcrp_watch = importlib.util.module_from_spec(SPEC)
sys.modules["hotcrp_watch"] = hotcrp_watch
SPEC.loader.exec_module(hotcrp_watch)


class TargetParsingTest(unittest.TestCase):
    def test_parse_targets_merges_paper_ids_and_urls(self):
        args = argparse.Namespace(
            site="https://iccad2026.hotcrp.com",
            paper="50, 51,50",
            url="https://iccad2026.hotcrp.com/paper/52, https://iccad2026.hotcrp.com/paper/51",
        )

        targets = hotcrp_watch.parse_targets(args)

        self.assertEqual([target.paper_id for target in targets], ["50", "51", "52"])
        self.assertEqual(targets[0].url, "https://iccad2026.hotcrp.com/paper/50")

    def test_parse_targets_requires_paper_or_url(self):
        args = argparse.Namespace(site="https://iccad2026.hotcrp.com", paper="", url="")

        with self.assertRaises(SystemExit):
            hotcrp_watch.parse_targets(args)


class HtmlParsingTest(unittest.TestCase):
    def test_extract_visible_text_ignores_hidden_nodes_and_scripts(self):
        html = """
        <html><body>
          <script>var score = 5;</script>
          <style>.x { display: none }</style>
          <input type="hidden" name="overall_merit" value="5">
          <div hidden>hidden recommendation 4</div>
          <div style="display:none">hidden confidence 3</div>
          <div>Comment: This is visible.</div>
        </body></html>
        """

        text = hotcrp_watch.extract_visible_text(html)

        self.assertIn("Comment: This is visible.", text)
        self.assertNotIn("score = 5", text)
        self.assertNotIn("hidden recommendation", text)
        self.assertNotIn("overall_merit", text)

    def test_scan_artifacts_finds_score_like_fields_with_context(self):
        html = """
        <html><body>
          <input type="hidden" name="overall_merit" value="4">
          <div data-confidence="3">Review A</div>
          <script>window.hotcrp = {"recommendation": "accept", "paperId": 50};</script>
          <!-- expertise: 2 -->
        </body></html>
        """

        artifacts = hotcrp_watch.scan_delivered_artifacts(html)
        fields = {(item["field"], item["location_type"]) for item in artifacts}

        self.assertIn(("overall_merit", "hidden_dom"), fields)
        self.assertIn(("data-confidence", "dom_attribute"), fields)
        self.assertIn(("recommendation", "inline_json"), fields)
        self.assertIn(("expertise", "html_comment"), fields)

    def test_parse_page_does_not_treat_paper_id_as_score(self):
        html = """
        <html><head><title>Paper 50</title></head><body>
          <h1>Paper 50</h1>
          <div class="comment">Comment: Needs more experiments.</div>
          <div>Submitted at 2026-06-10 01:02</div>
        </body></html>
        """

        parsed = hotcrp_watch.parse_page("50", "https://iccad2026.hotcrp.com/paper/50", html)

        self.assertEqual(parsed["paper_id"], "50")
        self.assertEqual(parsed["visible_score_fields"], [])
        self.assertEqual(parsed["delivered_artifacts"], [])
        self.assertTrue(parsed["visible_blocks"])

    def test_extract_page_messages_reports_hotcrp_errors(self):
        html = """
        <html><body>
          <div class="msg msg-error"><p>Email address not found.</p></div>
          <form id="f-signin"><input type="password" name="password"></form>
        </body></html>
        """

        messages = hotcrp_watch.extract_page_messages(html)

        self.assertEqual(messages, ["Email address not found."])


class SnapshotTest(unittest.TestCase):
    def test_changed_page_saves_snapshot_files_and_diff(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            target = hotcrp_watch.Target("50", "https://iccad2026.hotcrp.com/paper/50")
            old = hotcrp_watch.PageSnapshot(
                html="<html><body>Comment: old</body></html>",
                text="Comment: old",
                parsed={"paper_id": "50", "visible_blocks": ["Comment: old"]},
            )
            new = hotcrp_watch.PageSnapshot(
                html="<html><body>Comment: new Overall merit: 4</body></html>",
                text="Comment: new Overall merit: 4",
                parsed={"paper_id": "50", "visible_blocks": ["Comment: new Overall merit: 4"]},
            )

            result = hotcrp_watch.save_change_snapshot(out_dir, target, old, new, "20260610T010203Z")

            self.assertTrue(result.changed)
            self.assertTrue(result.snapshot_html.exists())
            self.assertTrue(result.snapshot_text.exists())
            self.assertTrue(result.snapshot_json.exists())
            self.assertTrue(result.diff_path.exists())
            self.assertIn("Overall merit", result.diff_path.read_text())
            self.assertEqual(json.loads(result.snapshot_json.read_text())["paper_id"], "50")

    def test_html_token_only_changes_do_not_count_as_page_changes(self):
        old = hotcrp_watch.PageSnapshot(
            html='<html><body><input name="post" value="oldtoken"><div>Comment: same</div></body></html>',
            text="Comment: same",
            parsed={
                "paper_id": "50",
                "raw_text_fingerprint": hotcrp_watch.stable_fingerprint("Comment: same"),
                "visible_blocks": ["Comment: same"],
                "visible_score_fields": [],
                "delivered_artifacts": [],
            },
        )
        new = hotcrp_watch.PageSnapshot(
            html='<html><body><input name="post" value="newtoken"><div>Comment: same</div></body></html>',
            text="Comment: same",
            parsed={
                "paper_id": "50",
                "raw_text_fingerprint": hotcrp_watch.stable_fingerprint("Comment: same"),
                "visible_blocks": ["Comment: same"],
                "visible_score_fields": [],
                "delivered_artifacts": [],
            },
        )

        self.assertFalse(hotcrp_watch.snapshots_changed(old, new))

    def test_same_block_count_with_changed_content_counts_as_change(self):
        old = hotcrp_watch.PageSnapshot(
            html="<html><body><div>Comment: needs clearer experiments</div></body></html>",
            text="Comment: needs clearer experiments",
            parsed=hotcrp_watch.parse_page(
                "50",
                "https://iccad2026.hotcrp.com/paper/50",
                "<html><body><div>Comment: needs clearer experiments</div></body></html>",
            ),
        )
        new = hotcrp_watch.PageSnapshot(
            html="<html><body><div>Comment: needs stronger baselines</div></body></html>",
            text="Comment: needs stronger baselines",
            parsed=hotcrp_watch.parse_page(
                "50",
                "https://iccad2026.hotcrp.com/paper/50",
                "<html><body><div>Comment: needs stronger baselines</div></body></html>",
            ),
        )

        self.assertEqual(len(old.parsed["visible_blocks"]), len(new.parsed["visible_blocks"]))
        self.assertTrue(hotcrp_watch.snapshots_changed(old, new))
        self.assertIn("visible_text", hotcrp_watch.snapshot_change_reasons(old, new))

    def test_save_login_failure_diagnostic_writes_html_and_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = hotcrp_watch.save_login_failure_diagnostic(
                Path(tmpdir),
                "<html><body><div class='msg msg-error'>Bad password</div></body></html>",
            )

            self.assertTrue(path.exists())
            self.assertIn("Bad password", path.read_text())
            self.assertIn("Bad password", (Path(tmpdir) / "login-failure.txt").read_text())


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self):
        self.calls = []
        self.posts = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse({"ok": True, "sessioninfo": {"postvalue": "abc123token"}})

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return FakeResponse({"code": 200, "msg": "success"})


class LoginTokenTest(unittest.TestCase):
    def test_fetch_session_post_value_uses_hotcrp_session_api(self):
        session = FakeSession()

        post_value = hotcrp_watch.fetch_session_post_value(session, "https://iccad2026.hotcrp.com")

        self.assertEqual(post_value, "abc123token")
        self.assertEqual(session.calls[0][0], "https://iccad2026.hotcrp.com/api/session")
        self.assertEqual(session.calls[0][1]["headers"]["Sec-Fetch-Site"], "same-origin")


class SafeSessionTest(unittest.TestCase):
    def test_safe_session_allows_only_signin_post(self):
        session = hotcrp_watch.SafeSession("https://iccad2026.hotcrp.com")

        with self.assertRaisesRegex(RuntimeError, "Refusing POST"):
            session.post("https://iccad2026.hotcrp.com/paper/50", data={"withdraw": "1"})

        self.assertIsNone(
            session.check_write_allowed("POST", "https://iccad2026.hotcrp.com/signin")
        )


class WatchResetTest(unittest.TestCase):
    def test_watch_reset_session_is_consumed_after_first_poll(self):
        args = argparse.Namespace(reset_session=True)

        self.assertTrue(hotcrp_watch.consume_reset_session(args))
        self.assertFalse(args.reset_session)
        self.assertFalse(hotcrp_watch.consume_reset_session(args))


class PushPlusTest(unittest.TestCase):
    def test_build_pushplus_payload_contains_change_paths(self):
        target = hotcrp_watch.Target("50", "https://iccad2026.hotcrp.com/paper/50")
        result = hotcrp_watch.SaveResult(
            changed=True,
            snapshot_html=Path("state/paper-50/snapshot.html"),
            snapshot_text=Path("state/paper-50/snapshot.txt"),
            snapshot_json=Path("state/paper-50/snapshot.json"),
            diff_path=Path("state/paper-50/diff.txt"),
            change_reasons=["visible_text"],
        )

        payload = hotcrp_watch.build_pushplus_payload("tok", target, result, "topic-a")

        self.assertEqual(payload["token"], "tok")
        self.assertEqual(payload["topic"], "topic-a")
        self.assertIn("HotCRP paper 50 changed", payload["title"])
        self.assertIn("Changes: visible_text", payload["content"])
        self.assertIn("diff.txt", payload["content"])
        self.assertEqual(payload["template"], "markdown")

    def test_send_pushplus_notification_posts_json(self):
        session = FakeSession()
        target = hotcrp_watch.Target("50", "https://iccad2026.hotcrp.com/paper/50")
        result = hotcrp_watch.SaveResult(
            changed=True,
            snapshot_html=Path("snapshot.html"),
            snapshot_text=Path("snapshot.txt"),
            snapshot_json=Path("snapshot.json"),
            diff_path=Path("diff.txt"),
        )

        hotcrp_watch.send_pushplus_notification("tok", target, result, session=session)

        self.assertEqual(session.posts[0][0], hotcrp_watch.DEFAULT_PUSHPLUS_URL)
        self.assertEqual(session.posts[0][1]["json"]["token"], "tok")


if __name__ == "__main__":
    unittest.main()
