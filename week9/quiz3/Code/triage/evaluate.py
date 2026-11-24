"""共用評估器：嚴格驗證（整段必須是合法 JSON + 符合 schema）與寬鬆驗證（可從文字中救出第一個 JSON）。"""
from __future__ import annotations

import json
import re
import statistics

from .paths import ROUTING_DATA
from .spec import INTENT2ROUTE, validate_triage

def load_benchmark():
    with (ROUTING_DATA / "benchmark.jsonl").open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def lenient_extract(raw: str) -> str | None:
    m = re.search(r"\{.*?\}", raw, re.S)
    return m.group(0) if m else None


def score_one(raw: str, gold: dict) -> dict:
    strict, why = validate_triage(raw)
    lenient = strict
    if strict is None:
        cand = lenient_extract(raw)
        if cand:
            lenient, _ = validate_triage(cand)
    tgt = strict or lenient
    return {
        "json_ok": _json_ok(raw), "schema_ok": strict is not None, "lenient_ok": lenient is not None, "fail": why,
        "route_ok": bool(strict and strict.route == gold["route"]),
        "intent_ok": bool(strict and strict.intent == gold["intent"]),
        "urgency_ok": bool(strict and strict.urgency == gold["urgency"]),
        "exact_ok": bool(strict and (strict.route, strict.intent, strict.urgency) == (gold["route"], gold["intent"], gold["urgency"])),
        "pred": None if tgt is None else [tgt.route, tgt.intent, tgt.urgency],
    }


def _json_ok(raw):
    try:
        json.loads(raw); return True
    except Exception:
        return False


def p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))] if xs else 0.0


def evaluate(node, cases, repeats: int = 1) -> dict:
    recs = []
    for _ in range(repeats):
        for c in cases:
            r = node.run(c["input"])
            s = score_one(r.raw, c["gold"])
            s.update(id=c["id"], kind=c["kind"], raw=r.raw, latency_ms=r.latency_ms, cost_usd=r.cost_usd)
            recs.append(s)
    n = len(recs)
    mean = lambda k: sum(1 for x in recs if x[k]) / n
    by_kind = {}
    for kd in ("single", "multi", "image"):
        sub = [x for x in recs if x["kind"] == kd]
        if sub:
            by_kind[kd] = sum(1 for x in sub if x["intent_ok"]) / len(sub)
    lat = [x["latency_ms"] for x in recs]
    valid = [x for x in recs if x["schema_ok"]]
    return {
        "node": node.name, "n": n,
        "json_ok": mean("json_ok"), "schema_ok": mean("schema_ok"), "lenient_ok": mean("lenient_ok"),
        "route_acc": mean("route_ok"), "intent_acc": mean("intent_ok"), "urgency_acc": mean("urgency_ok"), "exact": mean("exact_ok"),
        "intent_acc_given_valid": (sum(1 for x in valid if x["intent_ok"]) / len(valid)) if valid else 0.0,
        "intent_by_kind": by_kind,
        "lat_mean_ms": statistics.mean(lat), "lat_p95_ms": p95(lat),
        "cost_per_1k_usd": 1000 * statistics.mean(x["cost_usd"] for x in recs),
        "fail_reasons": {k: sum(1 for x in recs if x["fail"] == k) for k in {x["fail"] for x in recs if x["fail"]}},
        "records": recs,
    }
