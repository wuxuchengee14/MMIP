"""作業三的量化實驗：E0 級聯分流、E1 離題處理、E2 攻擊防禦、E3 UX（訂單沿用）。"""
import json, time, copy
import numpy as np
from pathlib import Path
from triage.evaluate import load_benchmark
from triage.guard import Guard
from triage.paths import CHAT_DATA
from triage.pipeline import Bot, Router, Session, DOMAIN_KW, build_default_bot
from triage.nodes.system_one import SystemOne
from triage.spec import BUSINESS_ROUTES, ORDER_ID_RE, INTENT2ROUTE

ROOT = Path(__file__).resolve().parents[1]; DATA = CHAT_DATA; RES = ROOT / "results"
sets = json.load(open(DATA / "eval_sets.json", encoding="utf-8"))
bench = load_benchmark()
sysone = SystemOne.load()
out = {}

def mk(**kw):
    return build_default_bot(**kw)

# ============================================================ E0：級聯分流 (System One → GRU+FSM → 澄清)
rows = []
for tau in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
    base = build_default_bot(router_kw={"tau_fast": tau}).router
    acc_n = acc_ok = unc = slow_calls = 0; lat = []; src = {}; agree_ok = agree_n = 0
    for c in bench:
        if c["gold"]["route"] in ("out_of_scope", "unsafe"): continue
        t0 = time.perf_counter(); r = base.classify(c["input"]); lat.append((time.perf_counter() - t0) * 1000)
        slow_calls += r["source"].startswith("D+C"); src[r["source"]] = src.get(r["source"], 0) + 1
        if r["source"] == "D+C(agree)": agree_n += 1; agree_ok += r["intent"] == c["gold"]["intent"]
        if r["uncertain"]: unc += 1
        else:
            acc_n += 1; acc_ok += r["intent"] == c["gold"]["intent"]
    n = acc_n + unc
    rows.append({"tau_fast": tau, "n": n, "answered": acc_n, "coverage": acc_n / n, "precision_on_answered": acc_ok / max(acc_n, 1),
                 "asked_clarify": unc, "sources": src, "rescued_by_slow_path": agree_n, "rescued_correct": agree_ok, "slow_path_rate": slow_calls / n, "lat_mean_ms": float(np.mean(lat)), "lat_p95_ms": float(np.percentile(lat, 95))})
    print("E0", {k: round(v, 3) if isinstance(v, float) else v for k, v in rows[-1].items()})
# 對照：只用 D（永遠作答）
dacc = np.mean([sysone.predict(c["input"])["intent"] == c["gold"]["intent"] for c in bench if c["gold"]["route"] not in ("out_of_scope", "unsafe")])
out["E0"] = {"rows": rows, "D_only_acc": float(dacc), "note": "tau_fast 預設 0.5 為事前設定，未在 Benchmark 上調整；此表僅供敏感度分析"}

# ============================================================ E1：離題處理
ood = sets["ood"]; indom = [c["input"] for c in bench if c["gold"]["route"] not in ("out_of_scope", "unsafe")]
def feat(t): p = sysone.predict(t); return p["route"] == "out_of_scope", p["confidence"], bool(DOMAIN_KW.search(t))
F_ood = [feat(t) for t in ood]; F_in = [feat(t) for t in indom]
def outcome(f, method, tau):
    """hard：直接導回服務範圍；soft：澄清(不建單)；accept：照業務路由處理"""
    oos, conf, kw = f
    if method == "M0": return "accept"                      # 基準線：封閉集合，永遠接受
    if oos: return "hard"
    if method == "M2" and conf < tau: return "hard"
    if method == "M3" and conf < tau and not kw: return "soft"
    return "accept"
def dist(F, m, tau):
    o = [outcome(f, m, tau) for f in F]; n = len(o)
    return {k: o.count(k) / n for k in ("hard", "soft", "accept")}
# 事前宣告的成本（產品假設）：離題被當業務處理=1；正常問題被硬拒=1；正常問題被澄清=0.3
def cost(Fo, Fi, m, tau):
    do, di = dist(Fo, m, tau), dist(Fi, m, tau)
    return do["accept"] * 1.0 + di["hard"] * 1.0 + di["soft"] * 0.3
dev = lambda L: [x for i, x in enumerate(L) if i % 2 == 0]; tst = lambda L: [x for i, x in enumerate(L) if i % 2 == 1]
grid = [round(x, 2) for x in np.arange(0.1, 0.95, 0.05)]
e1 = {"n_ood": len(ood), "n_indomain": len(indom), "cost_weights": {"ood_accepted": 1.0, "indomain_hard": 1.0, "indomain_soft": 0.3}, "methods": {}, "sweep": {}}
for m in ("M0", "M1", "M2", "M3"):
    tau = None
    if m in ("M2", "M3"):
        tau = min(grid, key=lambda t: (cost(dev(F_ood), dev(F_in), m, t), t))
    e1["methods"][m] = {"tau_chosen_on_dev": None if tau is None else float(tau),
        "test": {"ood": dist(tst(F_ood), m, tau), "indomain": dist(tst(F_in), m, tau)},
        "all": {"ood": dist(F_ood, m, tau), "indomain": dist(F_in, m, tau)}}
    print("E1", m, "tau", tau, "OOD(all)", {k: round(v, 2) for k, v in e1["methods"][m]["all"]["ood"].items()}, "IN(all)", {k: round(v, 2) for k, v in e1["methods"][m]["all"]["indomain"].items()})
for m in ("M2", "M3"):
    e1["sweep"][m] = [{"tau": float(t), "ood": dist(F_ood, m, t), "indomain": dist(F_in, m, t)} for t in grid]
# 離題 OOD 被誤路由到哪些業務路由（基準線）
cl = Router(sysone, None, closed_set=True)
e1["baseline_misroutes"] = {}
for t in ood:
    r = cl.classify(t); e1["baseline_misroutes"][r["route"]] = e1["baseline_misroutes"].get(r["route"], 0) + 1
# 對話層級：中途離題後能否接續完成任務
digress = ["今天天氣怎麼樣", "對了你喜歡貓還是狗", "幫我寫一首關於月亮的短詩", "台積電今天股價多少", "你會不會寂寞", "李白是哪個朝代的人", "歐元兌台幣匯率多少", "怎麼治療感冒"]
def dlg(bot):
    done = asks = 0
    for k, d in enumerate(digress):
        S = Session(); oid = f"B2026110{k+1}-0{k+1}00"[:14]
        oid = f"B20261101-{k+1:04d}"
        bot.respond(S, "我想退貨"); bot.respond(S, d); r = bot.respond(S, f"訂單編號是 {oid}")
        t = S.tickets[-1] if S.tickets else None
        done += bool(t and t.intent == "request_return" and t.order_id == oid); asks += S.n_questions
    return done, asks
b_base = build_default_bot(guard=None, ood_guard=False, router_kw={"closed_set": True}); b_new = build_default_bot()
e1["dialogue"] = {"n": len(digress), "baseline_completed": dlg(b_base)[0], "improved_completed": dlg(b_new)[0]}
print("E1 dialogue", e1["dialogue"]); out["E1"] = e1

# ============================================================ E2：攻擊防禦
def run_item(bot, text, img=None):
    S = Session()
    r = bot.respond(S, text, open(DATA / "images" / img, "rb").read() if img else None)
    route = (r.trace.get("triage") or {}).get("route")
    guard_blocked = bool(r.blocked)
    router_flag = route == "unsafe" or "router_unsafe" in (r.trace.get("guard") or {}).get("reasons", [])
    contained = guard_blocked or route in ("unsafe", "out_of_scope") or router_flag
    return {"contained": contained, "guard_blocked": guard_blocked, "router_flag": router_flag, "reasons": (r.trace.get("guard") or {}).get("reasons", [])}
def run_benign_case(bot, c):
    S = Session(); flagged = False
    for i, t in enumerate(c["turns"]):
        img = c["images"][0] if (c["images"] and i == len(c["turns"]) - 1) else None
        r = bot.respond(S, t, open(DATA / "images" / img, "rb").read() if img else None)
        route = (r.trace.get("triage") or {}).get("route")
        flagged |= bool(r.blocked) or route == "unsafe"
    return flagged
variants = {
    "baseline（無防禦層；僅路由器自帶的 unsafe 類別）": dict(guard=None),
    "L1 規則": dict(guard=Guard(layers=("L1",))),
    "L2 學習式分類器": dict(guard=Guard(layers=("L2",))),
    "L0+L1 正規化+規則": dict(guard=Guard(layers=("L0", "L1"))),
    "L0+L1+L2（文字防禦，不掃圖片）": dict(guard=Guard(), scan_images=False),
    "完整：L0+L1+L2 + 圖片 OCR 同樣檢查": dict(guard=Guard()),
}
at = sets["attack_text"]; benign_text = sets["benign_hard"]
benign_cases = [{"turns": [t], "images": []} for t in benign_text] + [c for c in bench if c["gold"]["route"] not in ("out_of_scope", "unsafe")]
e2 = {"n_attack_text": {k: len(v) for k, v in at.items()}, "n_attack_image": len(sets["attack_image"]), "n_benign": len(benign_cases), "variants": {}}
for name, kw in variants.items():
    bot = build_default_bot(**kw)
    res = {}
    for cat, items in at.items():
        rr = [run_item(bot, t) for t in items]
        res[cat] = {"contained": sum(x["contained"] for x in rr) / len(rr), "guard_detected": sum(x["guard_blocked"] for x in rr) / len(rr),
                    "missed": [t for t, x in zip(items, rr) if not x["contained"]]}
    rr = [run_item(bot, t, im) for im, t in sets["attack_image"]]
    res["image"] = {"contained": sum(x["contained"] for x in rr) / len(rr), "guard_detected": sum(x["guard_blocked"] for x in rr) / len(rr),
                    "missed": [im for (im, _), x in zip(sets["attack_image"], rr) if not x["contained"]]}
    tot = sum(len(v) for v in at.values()) + len(rr)
    allc = sum(res[c]["contained"] * len(at[c]) for c in at) + res["image"]["contained"] * len(rr)
    fl = [run_benign_case(bot, c) for c in benign_cases]
    fp_hard = float(np.mean(fl[:len(benign_text)])); fp_all = float(np.mean(fl))
    res["overall_contained"] = allc / tot; res["asr"] = 1 - allc / tot; res["benign_false_positive_hard"] = fp_hard; res["benign_false_positive_all"] = fp_all
    e2["variants"][name] = res
    print("E2", name, {k: round(v["contained"], 2) for k, v in res.items() if isinstance(v, dict)}, "ASR", round(res["asr"], 3), "FP(hard)", round(fp_hard, 3), "FP(all)", round(fp_all, 3), flush=True)
out["E2"] = e2

# ============================================================ E3：UX —— 訂單編號沿用（none / silent / confirm / implicit）
dialogs = json.load(open(DATA / "ux_dialogs.json", encoding="utf-8"))
def sim(bot, dialog):
    S = Session(); st = {"user_msgs": 0, "user_chars": 0, "wrong_first": 0, "wrong_final": 0, "ok": 0, "n_tasks": len(dialog["tasks"])}
    def say(m):
        st["user_msgs"] += 1; st["user_chars"] += len(m); return bot.respond(S, m)
    for task in dialog["tasks"]:
        n_before = len(S.tickets)
        reply = say(task["say"])
        for _ in range(6):
            if not reply.asked: break
            if S.awaiting == "confirm_order":
                msg = "是的，同一筆" if S.carry_candidate == task["order"] else f"不是，是 {task['order']}"
            elif S.awaiting == "order_id":
                msg = f"訂單編號是 {task['order']}"
            else:
                msg = task["say"]
            reply = say(msg)
        new = S.tickets[n_before:]
        for t in new:
            if task["order"] and t.order_id != task["order"]:
                st["wrong_first"] += 1
        if S.amendable and any(task["order"] and t.order_id != task["order"] for t in S.amendable):   # implicit：使用者看到沿用的訂單不對，更正
            say(f"不是這筆，是 {task['order']}")
        for t in new:
            st["ok"] += int(t.intent == task["route"] and t.order_id == task["order"])
            if task["order"] and t.order_id != task["order"]:
                st["wrong_final"] += 1
    st["questions"] = S.n_questions
    return st
e3 = {}
for mode in ("none", "silent", "confirm", "implicit"):
    tot = {}
    for d in dialogs:
        bot = build_default_bot(carry_over=mode, guard=None)
        for k, v in sim(bot, d).items(): tot[k] = tot.get(k, 0) + v
    n = tot["n_tasks"]
    tot.update(msgs_per_task=tot["user_msgs"] / n, chars_per_task=tot["user_chars"] / n, questions_per_task=tot["questions"] / n, task_success=tot["ok"] / n)
    e3[mode] = tot; print("E3", mode, {k: round(v, 3) if isinstance(v, float) else v for k, v in tot.items()})
out["E3"] = {"modes": e3, "n_dialogs": len(dialogs), "n_tasks": sum(len(d["tasks"]) for d in dialogs),
             "n_task_new_order": sum(1 for d in dialogs for i, t in enumerate(d["tasks"]) if t["order"] and i > 0 and any(x["order"] for x in d["tasks"][:i]) and t["order"] != next(x["order"] for x in reversed(d["tasks"][:i]) if x["order"]))}

# ============================================================ 快取
bot = build_default_bot(); rt = bot.router
texts = [c["input"] for c in bench]; import random
rng = random.Random(0); traffic = [rng.choice(texts[:20]) if rng.random() < 0.6 else rng.choice(texts) for _ in range(400)]
t0 = time.perf_counter(); [rt.classify(t) for t in traffic]; t_cache = time.perf_counter() - t0
rt2 = build_default_bot().router; rt2.cache_size = 0
t0 = time.perf_counter(); [rt2.classify(t) for t in traffic]; t_nocache = time.perf_counter() - t0
out["cache"] = {"hit_rate": rt.hits / rt.calls, "ms_per_req_cache": t_cache / 400 * 1000, "ms_per_req_nocache": t_nocache / 400 * 1000}
print("cache", out["cache"])
json.dump(out, open(RES / "exp3_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
