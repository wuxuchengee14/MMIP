"""Bounded Gemini retries; diagnostics deliberately exclude response messages and keys."""
from __future__ import annotations

import json
import math
import random
import re
import time
import urllib.error
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


def _seconds(value):
    try:
        number = float(str(value).removesuffix("s"))
        return number if math.isfinite(number) and number >= 0 else None
    except (TypeError, ValueError):
        return None


def http_diagnostic(error):
    """Read only safe, actionable metadata, never raw errors or request headers."""
    result = {"http_status": error.code, "quota": []}
    delays = []
    header = error.headers.get("Retry-After") if error.headers else None
    seconds = _seconds(header)
    if seconds is None and header:
        try:
            seconds = max(0, (parsedate_to_datetime(header) - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            pass
    if seconds is not None:
        delays.append(seconds)
    try:
        payload = json.loads(error.read(65536))
        data = payload.get("error", {})
        status = data.get("status", "")
        if isinstance(status, str) and re.fullmatch(r"[A-Z_]{1,80}", status):
            result["api_status"] = status
        for detail in data.get("details", []):
            if not isinstance(detail, dict):
                continue
            if str(detail.get("@type", "")).endswith("google.rpc.RetryInfo"):
                seconds = _seconds(detail.get("retryDelay"))
                if seconds is not None:
                    delays.append(seconds)
            if str(detail.get("@type", "")).endswith("google.rpc.QuotaFailure"):
                for violation in detail.get("violations", []):
                    if not isinstance(violation, dict):
                        continue
                    # Omit message, description, subject, dimensions and any key-bearing text.
                    quota = {}
                    for field in ("quotaMetric", "quotaId", "quotaValue"):
                        value = str(violation.get(field, ""))
                        if value and re.fullmatch(r"[A-Za-z0-9_./-]{1,240}", value):
                            quota[field] = value
                    if quota:
                        result["quota"].append(quota)
    except (ValueError, TypeError, AttributeError, OSError):
        pass
    finally:
        error.close()
    if delays:
        result["retry_after_seconds"] = max(delays)
    quotas = result["quota"]
    daily = any(re.search(r"per.?day|daily", str(q), re.I) for q in quotas)
    zero = any(q.get("quotaValue") == "0" for q in quotas)
    result["stop_until_quota_changes"] = error.code == 429 and (daily or zero)
    if result["stop_until_quota_changes"]:
        result["action"] = "Check AI Studio daily/zero quota; resume after reset or a quota change."
    elif error.code == 429:
        result["action"] = "Check request/token limits in AI Studio; quota type is unknown without details."
    elif error.code in (500, 502, 503, 504):
        result["action"] = "Server temporarily unavailable; resume later using the existing cache."
    else:
        result["action"] = "Check model availability, request parameters, key permissions and billing."
    return result


class GeminiRequestError(Exception):
    def __init__(self, diagnostic):
        super().__init__(diagnostic["action"])
        self.diagnostic = diagnostic


class RequestPolicy:
    def __init__(self, settings):
        # Environment override affects API pacing only, never model-training settings.
        import os
        self.interval = float(os.environ.get("GEMINI_INTERVAL_SECONDS", settings.get("gemini_interval_seconds", 15)))
        self.retries = int(settings.get("gemini_max_retries", 5))
        self.base = float(settings.get("gemini_retry_base_seconds", 10))
        self.cap = float(settings.get("gemini_retry_max_seconds", 120))
        self.max_server_wait = float(settings.get("gemini_max_server_wait_seconds", 300))
        if (self.retries < 0 or self.retries > 10 or
                any(not math.isfinite(x) or x < 0 for x in (self.interval, self.base, self.cap, self.max_server_wait))):
            raise ValueError("Invalid Gemini retry/pacing settings")
        self.next_request_at = 0.0

    def call(self, request):
        for attempt in range(self.retries + 1):
            remaining = self.next_request_at - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            try:
                return request()
            except urllib.error.HTTPError as error:
                diagnostic = http_diagnostic(error)
                retryable = error.code in (408, 429, 500, 502, 503, 504)
            except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as error:
                diagnostic = {"error_type": type(error).__name__,
                              "action": "Network failure or invalid judgment; resume using the existing cache."}
                retryable = True
            finally:
                # Applies to every attempt, including failures and subsequent images.
                self.next_request_at = time.monotonic() + self.interval
            diagnostic["attempts"] = attempt + 1
            hint = diagnostic.get("retry_after_seconds", 0)
            if (not retryable or diagnostic.get("stop_until_quota_changes") or
                    attempt == self.retries or hint > self.max_server_wait):
                print(f"Gemini stopped: HTTP {diagnostic.get('http_status', 'n/a')}; "
                      f"{diagnostic['action']} Attempts={attempt + 1}", flush=True)
                raise GeminiRequestError(diagnostic) from None
            backoff = min(self.cap, self.base * 2**attempt)
            wait = max(hint, self.interval, backoff + random.uniform(0, min(5, backoff * .2)))
            print(f"Gemini HTTP {diagnostic.get('http_status', 'n/a')}; "
                  f"retry {attempt + 1}/{self.retries} after {wait:.1f}s", flush=True)
            time.sleep(wait)
