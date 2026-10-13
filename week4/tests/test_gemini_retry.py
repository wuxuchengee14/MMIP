"""Offline regression tests for API limits, outages, key redaction and cache resume."""
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from common.utils import read_json, write_json
from quiz4.Code import gemini
from quiz4.Code.gemini_retry import GeminiRequestError, RequestPolicy, http_diagnostic


def http_error(code, details=None, headers=None):
    payload = {"error": {"status": "RESOURCE_EXHAUSTED" if code == 429 else "UNAVAILABLE",
                         "message": "SECRET_KEY must not appear in logs", "details": details or []}}
    return urllib.error.HTTPError("https://example.invalid", code, "failure", headers or {},
                                  io.BytesIO(json.dumps(payload).encode()))


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {"GEMINI_INTERVAL_SECONDS": "0"}))
        self.clock = 100.0
        def sleep(seconds):
            self.clock += seconds
        self.sleep = self.stack.enter_context(patch("quiz4.Code.gemini_retry.time.sleep", side_effect=sleep))
        self.stack.enter_context(patch("quiz4.Code.gemini_retry.time.monotonic", side_effect=lambda: self.clock))
        self.stack.enter_context(patch("quiz4.Code.gemini_retry.random.uniform", return_value=0))

    def test_503_recovery_uses_backoff(self):
        request = Mock(side_effect=[http_error(503), http_error(503), "ok"])
        self.assertEqual(RequestPolicy({}).call(request), "ok")
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [10, 20])

    def test_server_retry_info_and_header_are_respected(self):
        detail = {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "65s"}
        request = Mock(side_effect=[http_error(429, [detail], {"Retry-After": "70"}), "ok"])
        self.assertEqual(RequestPolicy({}).call(request), "ok")
        self.assertEqual(self.sleep.call_args.args[0], 70)

    def test_daily_quota_stops_without_retry_or_secret(self):
        detail = {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
            {"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
             "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quotaValue": "20"}]}
        request = Mock(side_effect=http_error(429, [detail]))
        with self.assertRaises(GeminiRequestError) as raised:
            RequestPolicy({}).call(request)
        self.assertTrue(raised.exception.diagnostic["stop_until_quota_changes"])
        self.assertNotIn("SECRET_KEY", json.dumps(raised.exception.diagnostic))
        self.assertEqual(request.call_count, 1)
        self.sleep.assert_not_called()

    def test_zero_quota_and_authentication_do_not_retry(self):
        detail = {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
            {"quotaId": "RequestsPerMinute", "quotaValue": "0"}]}
        for error in (http_error(429, [detail]), http_error(403)):
            request = Mock(side_effect=error)
            with self.assertRaises(GeminiRequestError):
                RequestPolicy({}).call(request)
            self.assertEqual(request.call_count, 1)

    def test_long_server_wait_stops_without_early_retry(self):
        request = Mock(side_effect=http_error(429, headers={"Retry-After": "3600"}))
        with self.assertRaises(GeminiRequestError):
            RequestPolicy({}).call(request)
        self.assertEqual(request.call_count, 1)
        self.sleep.assert_not_called()

    def test_bounded_outage_and_pacing_between_requests(self):
        request = Mock(side_effect=lambda: http_error(503))
        # Raise a fresh HTTPError because each body is consumed and closed.
        def fail():
            raise http_error(503)
        request.side_effect = fail
        with self.assertRaises(GeminiRequestError) as raised:
            RequestPolicy({"gemini_max_retries": 2}).call(request)
        self.assertEqual(raised.exception.diagnostic["attempts"], 3)
        with patch.dict(os.environ, {"GEMINI_INTERVAL_SECONDS": "15"}):
            policy = RequestPolicy({})
            policy.call(lambda: "first")
            policy.call(lambda: "second")
        self.assertEqual(self.sleep.call_args.args[0], 15)

    def test_malformed_error_body_still_provides_safe_diagnostic(self):
        error = urllib.error.HTTPError("https://example.invalid", 503, "SECRET_KEY", {}, io.BytesIO(b"bad json"))
        result = http_diagnostic(error)
        self.assertEqual(result["http_status"], 503)
        self.assertNotIn("SECRET_KEY", json.dumps(result))

    def test_cache_survives_outage_and_resume_completes(self):
        judgment = {"verdict": "partial", "score": 3, "subject_correct": True,
                    "action_correct": False, "scene_correct": True, "reason": "Offline fixture"}
        with tempfile.TemporaryDirectory() as tmp, patch.object(gemini, "ROOT", Path(tmp)), \
             patch.dict(os.environ, {"GEMINI_API_KEY": "FAKE_NOT_A_REAL_KEY", "GEMINI_MODEL": "test-model"}):
            root = Path(tmp)
            folder = root / "quiz1/Data/flickr8k"
            folder.mkdir(parents=True)
            rows = []
            for i in range(2):
                sid = f"{i}.jpg"
                Image.new("RGB", (8, 8), (i * 100, 40, 40)).save(folder / sid)
                rows.append({"id": sid, "image": sid, "caption": "a cat"})
            for kind in ("lstm", "transformer"):
                write_json(root / f"quiz4/Output/{kind}/greedy/predictions.json", rows)
            config = {"seed": 42, "caption": {"gemini_model": "test-model", "gemini_samples": 2,
                                               "gemini_max_retries": 1}}
            def outage(*args):
                raise http_error(503)
            calls = 0
            def first_run(*args):
                nonlocal calls
                calls += 1
                return (judgment, {}) if calls == 1 else outage()
            with patch.object(gemini, "request_judgment", side_effect=first_run):
                self.assertFalse(gemini.run(config))
            self.assertEqual(calls, 3, "Stop after first exhausted image, rather than hammering remaining images")
            output = root / "quiz4/Output/gemini"
            self.assertEqual(read_json(output / "summary.json")["successful"], 1)
            self.assertEqual(read_json(output / "errors.json")[0]["attempts"], 2)
            with patch.object(gemini, "request_judgment", return_value=(judgment, {})) as request:
                self.assertTrue(gemini.run(config))
                self.assertEqual(request.call_count, 3, "Previously successful image must not be requested again")
                self.assertTrue(gemini.run(config))
                self.assertEqual(request.call_count, 3)
            self.assertEqual(read_json(output / "summary.json")["successful"], 4)
            self.assertEqual(read_json(output / "errors.json"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
