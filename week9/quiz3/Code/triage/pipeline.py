"""作業三：多模態、多輪、具分流能力的客服機器人（核心邏輯，不含 UI）。

處理流程（對應 quiz1/Output/architecture.png）：
  使用者訊息(+圖片)
    → Guard(L0 正規化/L1 規則/L2 分類器；圖片 OCR 文字同樣檢查)       被攔截 → 安全回覆，且「不更新對話狀態」
    → Perception(圖片品質檢查 + OCR + 實體抽取)                       品質不佳 → 請使用者重拍，保留待辦
    → Context Keeper(槽位記憶、待補欄位、話題緩衝區)
    → Router 級聯：System One(快) → 低信心時 GRU+FSM(慢) → 仍不確定則澄清
    → OOD 判斷(離題)                                                    → 導回服務範圍；連續離題 → 轉真人
    → Policy(必填槽位、需要照片、訂單編號沿用/確認)                     → 追問 or 建立 HandoffTicket
    → Replier(模板回覆 + quick replies + 話題銜接)
"""
from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

from .context import build_input
from .guard import Guard, sanitize_for_model
from .spec import (BUSINESS_ROUTES, DEPARTMENT, INTENT2ROUTE, NEEDS_IMAGE, ORDER_ID_RE, REQUIRED_SLOTS, HandoffTicket,
                   ImageFinding, validate_triage)
from .vision import analyze_image

INTENT_ZH = {
    "track_shipment": "查詢物流", "modify_order": "修改訂單", "cancel_order": "取消訂單", "request_return": "申請退換貨",
    "report_defect": "回報瑕疵", "refund_status": "查詢退款", "troubleshoot_device": "故障排除", "error_code": "錯誤代碼",
    "invoice_request": "發票服務", "payment_issue": "付款問題", "spec_question": "商品規格", "stock_check": "庫存查詢",
    "login_problem": "登入問題", "update_profile": "會員資料", "complaint": "客訴", "request_human": "真人客服",
}
ROUTE_ZH = {"order_logistics": "訂單物流", "return_refund": "退換貨", "tech_support": "技術支援", "billing_invoice": "發票付款",
            "product_inquiry": "商品諮詢", "account": "會員帳號", "human_agent": "真人客服"}
SLA = {"high": "30 分鐘", "normal": "4 小時", "low": "1 個工作天"}
NEXT_SUGGEST = {  # 話題銜接：處理完一件事後，最常接著問的事（規則表，可由真實對話統計取代）
    "track_shipment": ["修改收件資訊", "開立發票"], "modify_order": ["查詢物流", "取消訂單"], "cancel_order": ["查詢退款進度", "重新下單諮詢"],
    "request_return": ["查詢退款進度", "開立折讓/發票"], "report_defect": ["申請換貨", "查詢退款進度"], "refund_status": ["開立發票", "查詢物流"],
    "invoice_request": ["查詢物流", "付款問題"], "payment_issue": ["查詢訂單狀態", "開立發票"],
}
DOMAIN_KW = re.compile("訂單|退貨|退款|換貨|發票|統編|付款|刷卡|扣款|帳號|登入|密碼|驗證碼|客服|出貨|物流|配送|包裹|收件|貨|保固|現貨|庫存|補貨|規格|"
                       "相容|故障|壞|當機|開不了機|錯誤|代碼|連不上|充電|瑕疵|刮傷|破損|優惠券|會員|地址|電話號碼|取消|客訴|投訴|真人|人工")
YES = re.compile(r"^(是|對|好|沒錯|嗯|可以|ok|yes|y|沿用|就是|是的|對的|好的)", re.I)
NO = re.compile(r"(不是|不對|否|不用|換一|另一|別的|no\b)", re.I)
FIX = re.compile(r"(不是這|不是同一|不對|錯了|搞錯|另一筆|別筆|不是這筆)")
CLOSING = re.compile(r"^(沒有了|沒了|不用了|沒有其他|謝謝|感謝|掰掰|再見|bye|thanks|ok謝謝|沒事了)", re.I)
FILLER = re.compile(r"[\s，。,.!！？?：:（）()「」]|訂單編號|訂單|單號|編號|是|我的|謝謝|感謝|麻煩|請|就是|這個|這張|這筆|如下|如圖|是的|沿用")


def is_slot_only(text: str) -> bool:
    return len(FILLER.sub("", ORDER_ID_RE.sub("", text))) <= 2


@dataclass
class Session:
    history: list = field(default_factory=list)          # [(role, text)]
    topic_turns: list = field(default_factory=list)      # 目前話題的使用者訊息（處理完一張工單後清空）
    topic_ocr: list = field(default_factory=list)
    order_id: Optional[str] = None                       # 目前話題已確認的槽位
    last_order_id: Optional[str] = None                  # 上一個已處理話題用過的訂單（carry-over 候選）
    pending: list = field(default_factory=list)          # 待辦 [{route,intent,urgency,confidence}]
    awaiting: Optional[str] = None                       # order_id | image | confirm_order | clarify
    carry_candidate: Optional[str] = None
    images: list = field(default_factory=list)           # 本話題已收到的 ImageFinding
    has_image_evidence: bool = False
    strikes: int = 0                                     # 連續離題次數
    tickets: list = field(default_factory=list)
    n_questions: int = 0                                 # 機器人追問使用者的次數（UX 指標）
    seq: int = 0
    carried: bool = False                                # 本話題的訂單是否沿用自上一筆
    amendable: list = field(default_factory=list)        # 以沿用訂單建立、仍可更正的工單


@dataclass
class Reply:
    text: str
    quick_replies: list = field(default_factory=list)
    triage: Optional[dict] = None
    tickets: list = field(default_factory=list)
    blocked: bool = False
    asked: bool = False
    trace: dict = field(default_factory=dict)


class Router:
    """級聯：System One 快速分流；信心不足才叫較慢的生成式節點（GRU+FSM；正式環境可換成 VLM/LLM + Structured Outputs）。"""

    def __init__(self, sysone, slow_node=None, tau_fast=0.50, tau_ood=0.30, cache_size=512, closed_set=False):
        self.d, self.slow, self.tau_fast, self.tau_ood = sysone, slow_node, tau_fast, tau_ood
        self.closed_set = closed_set   # True：基準線，只能在 7 個業務路由裡選（無離題/注入類別）
        self.cache: OrderedDict = OrderedDict(); self.cache_size = cache_size
        self.hits = self.calls = 0

    def classify(self, text: str) -> dict:
        self.calls += 1
        if text in self.cache:
            self.hits += 1; self.cache.move_to_end(text)
            return {**self.cache[text], "source": self.cache[text]["source"] + "(cache)"}
        p = self.d.predict(text)
        if self.closed_set:
            from .spec import INTENT2ROUTE as _I2R
            biz = {i: v for i, v in p["p_intent"].items() if _I2R[i] in BUSINESS_ROUTES}
            bi = max(biz, key=biz.get)
            p = {**p, "intent": bi, "route": _I2R[bi], "confidence": biz[bi], "p_intent": biz}
        top = sorted(p["p_intent"].items(), key=lambda kv: -kv[1])[:3]
        out = {"route": p["route"], "intent": p["intent"], "urgency": p["urgency"], "confidence": p["confidence"],
               "source": "D", "top3": top, "uncertain": False}
        if p["confidence"] < self.tau_fast and self.slow is not None and p["route"] not in ("out_of_scope", "unsafe"):
            raw = self.slow.run(text).raw
            o, _ = validate_triage(raw)
            if o is not None and o.intent == p["intent"]:
                out["source"] = "D+C(agree)"               # 兩個獨立節點同意 → 接受（信心維持 D 的機率，不灌水）
            else:
                out["source"] = "D+C(disagree)"; out["uncertain"] = True
        elif p["confidence"] < self.tau_fast:
            out["uncertain"] = True
        self.cache[text] = out
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return out

    def is_offtopic(self, text: str, r: dict) -> bool:
        if r["route"] == "out_of_scope":
            return True
        return r["confidence"] < self.tau_ood and not DOMAIN_KW.search(text)


class Bot:
    def __init__(self, router: Router, guard: Optional[Guard] = None, ood_guard=True, carry_over="confirm", bridging=True,
                 multi_intent=True, scan_images=True):
        self.router, self.guard, self.ood_guard = router, guard, ood_guard
        self.carry_over, self.bridging, self.multi_intent, self.scan_images = carry_over, bridging, multi_intent, scan_images

    # ------------------------------------------------------------------ 工具
    def _ticket(self, S: Session, item: dict, action: str, missing=None) -> HandoffTicket:
        S.seq += 1
        summ = sanitize_for_model(" / ".join(S.topic_turns))[:160]
        t = HandoffTicket(ticket_id=f"T{time.strftime('%y%m%d')}-{len(S.tickets) + 1:03d}", department=DEPARTMENT.get(item["route"], "-"),
                          route=item["route"], intent=item["intent"], urgency=item["urgency"], confidence=round(float(item["confidence"]), 2),
                          order_id=S.order_id, image_findings=list(S.images), missing_slots=missing or [], next_action=action,
                          conversation_summary=summ)
        S.tickets.append(t)
        return t

    def _quick_menu(self):
        return ["查詢物流", "申請退貨", "發票問題", "技術支援", "轉真人客服"]

    def _blocked(self, S, text, reasons, trace, where="text"):
        S.history.append(("user", "[已攔截的輸入]"))
        msg = "這個要求我沒辦法處理。如果您有訂單、退換貨、發票或商品相關的問題，直接告訴我就可以了。"
        trace["guard"] = {"blocked": True, "where": where, "reasons": reasons}
        return Reply(msg, ["查詢物流", "申請退貨", "轉真人客服"], blocked=True, trace=trace)

    def _pending_digression(self, session, text, image, trace):
        # Classify a sudden change of topic on its own so pending service history
        # cannot make a weather question look like a continuation of a return.
        if not self.ood_guard or not session.pending or image is not None or not text.strip():
            return None
        if ORDER_ID_RE.search(text) or DOMAIN_KW.search(text):
            return None
        if session.awaiting == 'confirm_order' and (YES.match(text) or NO.search(text)):
            return None
        prediction = self.router.d.predict(text)
        if prediction['route'] != 'out_of_scope' and prediction['confidence'] >= self.router.tau_ood:
            return None
        session.strikes += 1
        trace['offtopic_strikes'] = session.strikes
        trace['pending_preserved'] = True
        reminder = {
            'order_id': '請提供訂單編號或上傳訂單截圖。',
            'image': '請上傳商品或外箱的照片。',
            'confirm_order': f'請確認是否沿用訂單 {session.carry_candidate}。',
        }.get(session.awaiting, '可以繼續補充剛才的問題。')
        task = INTENT_ZH[session.pending[0]['intent']]
        message = f'我可以協助商城客服問題。剛才的「{task}」仍在等候補件。{reminder}'
        quick = ['是，同一筆', '不是，換一筆'] if session.awaiting == 'confirm_order' else []
        if session.strikes >= 3:
            message += '如果需要，也可以轉真人客服。'
            quick.append('轉真人客服')
        return Reply(message, quick, asked=True, trace=trace)

    # ------------------------------------------------------------------ 主入口
    def respond(self, S: Session, text: str, image: bytes | None = None) -> Reply:
        t0 = time.perf_counter(); trace: dict = {}
        text = text or ""
        clean = text
        if self.guard:
            g = self.guard.check(text)
            clean = g.clean_text
            trace["guard"] = {"blocked": g.blocked, "reasons": g.reasons, "score": round(g.score, 2)}
            if g.blocked:
                return self._blocked(S, text, g.reasons, trace)
        digression = self._pending_digression(S, clean, image, trace)
        if digression is not None:
            return digression
        finding: Optional[ImageFinding] = None
        if image is not None:
            ts = time.perf_counter(); finding = analyze_image(image); trace["perception_ms"] = round((time.perf_counter() - ts) * 1000)
            trace["image"] = {"doc_type": finding.doc_type, "quality_ok": finding.quality_ok, "ocr": finding.ocr_text[:80]}
            if not finding.quality_ok:
                S.history.append(("user", clean + " [圖片品質不佳]"))
                if clean.strip():
                    S.topic_turns.append(clean)                  # 文字內容保留，補件後仍能判斷意圖
                S.n_questions += 1
                return Reply(f"這張照片{finding.quality_note}，我讀不到裡面的內容。可以在光線充足的地方重新拍一張，或直接輸入訂單編號嗎？",
                             ["重新上傳", "改用文字描述"], asked=True, trace=trace)
            if self.guard and self.scan_images:
                gi = self.guard.check(finding.ocr_text)
                if gi.blocked:
                    return self._blocked(S, text, gi.reasons, trace, where="image")
            S.images.append(finding); S.has_image_evidence = True
            S.topic_ocr.append(finding.ocr_text)

        S.history.append(("user", clean or "[圖片]"))
        if CLOSING.match(clean.strip()) and not S.pending:
            S.strikes = 0
            return Reply("不客氣！有任何問題隨時再找我。祝您有美好的一天 😊", [], trace=trace)

        # ---- 槽位
        ids = ORDER_ID_RE.findall(clean) + (finding.order_ids if finding else [])
        if ids:
            S.order_id = ids[0]

        # ---- 更正「已沿用但其實不對」的訂單（implicit 模式）
        if S.amendable:
            wants_fix = bool(FIX.search(clean))
            if ids and (wants_fix or is_slot_only(clean) or S.awaiting == "amend_order"):
                for t in S.amendable:
                    t.order_id = ids[0]
                tid = "、".join(t.ticket_id for t in S.amendable)
                S.last_order_id = ids[0]; S.amendable = []; S.awaiting = None
                S.order_id = None                      # 更正是針對「上一張工單」，不是新話題的槽位
                return Reply(f"已把工單 {tid} 更正為訂單 {ids[0]}，謝謝您的確認。", [], trace=trace)
            if wants_fix or S.awaiting == "amend_order":
                S.awaiting = "amend_order"; S.n_questions += 1
                return Reply("好的，請告訴我正確的訂單編號。", [], asked=True, trace=trace)
            S.amendable = []

        # ---- 澄清「是否沿用上一筆訂單」
        awaiting0 = S.awaiting
        if awaiting0 == "confirm_order":
            if ids:
                S.awaiting = None
            elif NO.search(clean) and not YES.match(clean):
                S.awaiting = "order_id"; S.carry_candidate = None; S.n_questions += 1
                return Reply("了解，請提供這次要處理的訂單編號（例如 A20261104-0087），或上傳訂單截圖。", [], asked=True, trace=trace)
            elif YES.match(clean):
                S.order_id = S.carry_candidate; S.awaiting = None
            else:
                S.awaiting = None; S.carry_candidate = None  # 使用者換話題了，放棄確認

        resume = bool(S.pending) and (
            (awaiting0 == "confirm_order" and bool(S.order_id))
            or (awaiting0 == "order_id" and bool(ids) and is_slot_only(clean))
            or (awaiting0 == "image" and finding is not None)
            or (awaiting0 in ("order_id", "image") and finding is not None and not clean.strip()))
        if resume:
            trace["route_source"] = "pending-resume"
            return self._advance(S, trace, t0)

        # ---- 分流
        S.topic_turns.append(clean)
        items = self._classify_items(S, clean, trace)
        if items is None:                                   # 不確定
            S.awaiting = "clarify"; S.n_questions += 1
            cands = [INTENT_ZH.get(i, i) for i, _ in trace["triage_top3"] if i in INTENT_ZH][:3]
            msg = "我想確認一下您的需求，您是想要：" + "、".join(cands) + " 嗎？也可以多描述一點，或附上截圖。"
            if trace.get("soft_ood"):
                msg = "我不太確定這是不是我能協助的問題。如果和訂單、退換貨、發票、商品或帳號有關，請再多說一點；您是想：" + "、".join(cands) + " 嗎？"
            return Reply(msg, cands + ["不是，換個話題"], triage=trace.get("triage"), asked=True, trace=trace)
        if items == "offtopic":
            S.topic_turns.pop()
            S.strikes += 1
            trace["offtopic_strikes"] = S.strikes
            if S.strikes >= 3:
                return Reply("看起來您想聊的和我能協助的範圍不太一樣。需要我幫您轉接真人客服嗎？", ["轉真人客服", "查詢物流", "申請退貨"],
                             triage=trace.get("triage"), trace=trace)
            tail = ""
            if S.pending:
                tail = f"\n我們剛剛還在處理「{INTENT_ZH[S.pending[0]['intent']]}」，" + ("請提供訂單編號就能繼續喔。" if S.awaiting == "order_id" else "回覆我就能繼續。")
            return Reply("這個問題超出我能協助的範圍了 😅 我可以幫您處理：訂單物流、退換貨、技術支援、發票付款、商品諮詢、會員帳號。" + tail,
                         self._quick_menu(), triage=trace.get("triage"), trace=trace)
        S.strikes = 0
        S.pending = items
        return self._advance(S, trace, t0)

    # ------------------------------------------------------------------ 分流（含多意圖拆句）
    def _classify_items(self, S, clean, trace):
        model_in = sanitize_for_model(build_input(S.topic_turns, S.topic_ocr)) if self.guard else build_input(S.topic_turns, S.topic_ocr)
        ts = time.perf_counter()
        r = self.router.classify(model_in)
        trace["router_ms"] = round((time.perf_counter() - ts) * 1000, 2)
        trace["triage"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("top3",)}
        trace["triage_top3"] = r["top3"]
        if self.ood_guard:
            if r["route"] == "out_of_scope":
                return "offtopic"                           # 硬離題：分類器有把握是離題
            if r["confidence"] < self.router.tau_ood and not DOMAIN_KW.search(clean):
                trace["soft_ood"] = True
                return None                                 # 軟離題：不確定是不是我們的範圍 → 澄清，不建單
        if not self.ood_guard and r["route"] == "out_of_scope":
            return "offtopic"
        if r["route"] == "unsafe":                           # 路由器本身認為是注入（第二道保險）
            trace["guard"] = {**trace.get("guard", {}), "blocked": True, "reasons": ["router_unsafe"]}
            return "offtopic"
        if r["uncertain"]:
            return None
        items = [{"route": r["route"], "intent": r["intent"], "urgency": r["urgency"], "confidence": r["confidence"]}]
        if self.multi_intent and len(S.topic_turns) == 1:
            extra = self._split_intents(clean, r)
            if extra:
                items += extra; trace["multi_intent"] = [i["intent"] for i in items]
        return items

    def _split_intents(self, clean, first):
        """多意圖：把一句話拆成子句分別分流；只有「至少兩個不同且高信心的業務意圖」才拆。"""
        parts = [p for p in re.split(r"[，,；;。]|另外|還有|以及|並且|而且|順便|同時", clean) if len(p.strip()) >= 4]
        if len(parts) < 2:
            return []
        found = {}
        for p in parts:
            q = self.router.d.predict(p)
            if q["confidence"] >= 0.45 and q["route"] in BUSINESS_ROUTES:
                found.setdefault(q["intent"], {"route": q["route"], "intent": q["intent"], "urgency": q["urgency"], "confidence": q["confidence"]})
        if len(found) < 2 or first["intent"] not in found:
            return []
        found.pop(first["intent"])
        return list(found.values())[:2]

    # ------------------------------------------------------------------ Policy：槽位檢查 → 追問或建單
    def _advance(self, S: Session, trace, t0) -> Reply:
        item = S.pending[0]
        if item["route"] == "human_agent":
            return self._finish(S, trace, t0, action="escalate_human")
        need = []
        for it in S.pending:
            for s in REQUIRED_SLOTS.get(it["intent"], []):
                if s == "order_id" and not S.order_id and "order_id" not in need:
                    need.append("order_id")
        if "order_id" in need and S.last_order_id and S.awaiting != "order_id" and self.carry_over != "none":
            if self.carry_over in ("silent", "implicit"):
                S.order_id = S.last_order_id; need.remove("order_id"); S.carried = self.carry_over == "implicit"
                trace["carry_over"] = self.carry_over
            else:
                S.awaiting = "confirm_order"; S.carry_candidate = S.last_order_id; S.n_questions += 1
                trace["carry_over"] = "ask"
                return Reply(f"好的，我來協助您處理「{INTENT_ZH[item['intent']]}」。請問是剛剛那筆訂單 {S.last_order_id} 嗎？",
                             ["是，同一筆", "不是，另一筆"], triage=trace.get("triage"), asked=True, trace=trace)
        if "order_id" in need:
            S.awaiting = "order_id"; S.n_questions += 1
            return Reply(f"好的，我來協助您處理「{INTENT_ZH[item['intent']]}」。請提供訂單編號（例如 A20261104-0087），或直接上傳訂單截圖。",
                         [], triage=trace.get("triage"), asked=True, trace=trace)
        if any(it["intent"] in NEEDS_IMAGE for it in S.pending) and not S.has_image_evidence:
            S.awaiting = "image"; S.n_questions += 1
            return Reply("了解，很抱歉讓您遇到這個狀況。請拍一張商品或外箱受損處的照片上傳，我會一起交給同仁判斷。", [],
                         triage=trace.get("triage"), asked=True, trace=trace)
        return self._finish(S, trace, t0, action="handoff")

    def _finish(self, S, trace, t0, action) -> Reply:
        tickets = [self._ticket(S, it, action) for it in S.pending]
        names = "、".join(f"「{DEPARTMENT[t.route]}」" for t in tickets)
        ids_txt = "、".join(t.ticket_id for t in tickets)
        urg = SLA[tickets[0].urgency]
        if action == "escalate_human":
            text = f"我已經幫您轉接真人客服（工單 {ids_txt}），專員會在 {urg} 內與您聯繫。造成不便非常抱歉。"
        else:
            text = f"已為您建立工單 {ids_txt}，轉交{names}處理，預計 {urg} 內回覆。"
            if tickets[0].order_id:
                text += f"（訂單 {tickets[0].order_id}）"
        quick = []
        if self.bridging and action == "handoff":
            sug = NEXT_SUGGEST.get(tickets[0].intent, [])
            text += "\n還有其他我能幫上忙的嗎？" + ("例如：" + "、".join(sug) if sug else "")
            quick = sug + ["沒有了，謝謝"]
        if S.carried:
            S.amendable = list(tickets)
            text += f"\n（已沿用訂單 {S.order_id}；如果不是這筆，請直接告訴我正確的訂單編號）"
            quick = ["不是這筆"] + quick
            S.carried = False
        S.last_order_id = S.order_id or S.last_order_id
        S.order_id = None; S.pending = []; S.awaiting = None; S.topic_turns = []; S.topic_ocr = []
        S.images = []; S.has_image_evidence = False; S.carry_candidate = None
        trace["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return Reply(text, quick, triage=trace.get("triage"), tickets=[t.model_dump() for t in tickets], trace=trace)


def build_default_bot(**kw) -> Bot:
    from pathlib import Path
    from .nodes.sft_node import CondGRU, GRUNode, load_bundle
    from .nodes.system_one import MODEL_PATH, SystemOne
    from .nodes.fsm import JsonFSM
    root = Path(__file__).resolve().parents[1]
    sysone = SystemOne.load(MODEL_PATH)
    B = load_bundle(root / "results" / "bundle_b.pkl")
    D, H, E = B["dims"]
    m = CondGRU(B["vocab"], D, H, E); m.p = B["B2"]
    slow = GRUNode(m, B["vec"], "C2_slow", constrained=True, fsm=JsonFSM())
    guard = kw.pop("guard", Guard())
    return Bot(Router(sysone, slow, **kw.pop("router_kw", {})), guard, **kw)
