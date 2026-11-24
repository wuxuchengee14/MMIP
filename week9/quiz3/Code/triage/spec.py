"""作業一：結構化輸出 Spec（單一事實來源）。

業務情境：3C 電商客服（EriXNet 商城）。使用者分多輪描述問題，常附截圖／照片；
機器人確認需求後，把案件分流給下游部門。

兩層 Spec：
  1. TriageOutput  —— 「分流節點」（A/B/C/D 四種實作）必須輸出的結構。刻意保持小而扁平，
                       讓 SFT / Constrained Decoding / 小模型都做得到。
  2. HandoffTicket —— 分流後交給下游部門的工單，由 pipeline 以「已驗證欄位 + 規則」組出，
                       不直接信任模型自由文字。
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------- 路由與意圖
ROUTE_INTENTS: dict[str, list[str]] = {
    "order_logistics": ["track_shipment", "modify_order", "cancel_order"],
    "return_refund": ["request_return", "report_defect", "refund_status"],
    "tech_support": ["troubleshoot_device", "error_code"],
    "billing_invoice": ["invoice_request", "payment_issue"],
    "product_inquiry": ["spec_question", "stock_check"],
    "account": ["login_problem", "update_profile"],
    "human_agent": ["complaint", "request_human"],
    "out_of_scope": ["chitchat_offtopic"],
    "unsafe": ["prompt_injection"],
}
ROUTES: list[str] = list(ROUTE_INTENTS)
INTENTS: list[str] = [i for r in ROUTES for i in ROUTE_INTENTS[r]]
INTENT2ROUTE: dict[str, str] = {i: r for r, v in ROUTE_INTENTS.items() for i in v}
URGENCIES: list[str] = ["low", "normal", "high"]
BUSINESS_ROUTES: list[str] = [r for r in ROUTES if r not in ("out_of_scope", "unsafe")]

DEPARTMENT = {
    "order_logistics": "訂單與物流組",
    "return_refund": "退換貨組",
    "tech_support": "技術支援組",
    "billing_invoice": "帳務與發票組",
    "product_inquiry": "商品諮詢組",
    "account": "會員服務組",
    "human_agent": "真人客服（升級）",
}

# 這些 intent 沒有訂單編號就無法處理 → 由 policy 層追問，而不是讓模型亂猜
REQUIRED_SLOTS: dict[str, list[str]] = {
    "track_shipment": ["order_id"], "modify_order": ["order_id"], "cancel_order": ["order_id"],
    "request_return": ["order_id"], "report_defect": ["order_id"], "refund_status": ["order_id"],
    "payment_issue": ["order_id"], "invoice_request": ["order_id"],
}
NEEDS_IMAGE = {"report_defect"}  # 瑕疵回報需要照片佐證

ORDER_ID_RE = re.compile(r"[A-Z]\d{8}-\d{4}")  # 例：A20261104-0087
CONF_RE = re.compile(r"^(0\.\d{2}|1\.0)$")


# ---------------------------------------------------------------- 分流節點輸出
class TriageOutput(BaseModel):
    """分流節點輸出。範例：
    {"route":"return_refund","intent":"report_defect","urgency":"high","confidence":0.95}
    """
    model_config = ConfigDict(extra="forbid", strict=True)

    route: Literal[tuple(ROUTES)]  # type: ignore[valid-type]
    intent: Literal[tuple(INTENTS)]  # type: ignore[valid-type]
    urgency: Literal["low", "normal", "high"]
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _route_intent_consistent(self):
        if INTENT2ROUTE[self.intent] != self.route:
            raise ValueError(f"intent {self.intent} 不屬於 route {self.route}")
        return self


def triage_json_schema() -> dict:
    """給 OpenAI/Gemini 類 Structured Outputs 或 outlines 使用的 JSON Schema。"""
    s = TriageOutput.model_json_schema()
    s["additionalProperties"] = False
    s["allOf"] = [
        {"if": {"properties": {"route": {"const": route}}},
         "then": {"properties": {"intent": {"enum": intents}}}}
        for route, intents in ROUTE_INTENTS.items()
    ]
    return s


def validate_triage(text: str) -> tuple[Optional[TriageOutput], str]:
    """嚴格驗證：必須整段就是合法 JSON，且符合 schema。回傳 (物件|None, 失敗原因)。"""
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate key: {key}")
            result[key] = value
        return result

    def reject_non_json_constant(token):
        raise ValueError(f"Invalid JSON constant: {token}")

    try:
        obj = json.loads(text, object_pairs_hook=unique_keys, parse_constant=reject_non_json_constant)
    except (ValueError, TypeError):
        return None, "json_parse"
    try:
        return TriageOutput.model_validate(obj), ""
    except Exception:
        return None, "schema"


def canonical(route: str, intent: str, urgency: str, confidence: float) -> str:
    """固定格式的輸出字串（SFT 目標、FSM 語言都以此為準）。confidence 為 0.00~0.99 兩位小數，或 1.0。"""
    c = "1.0" if confidence >= 0.995 else f"{confidence:.2f}"
    return f'{{"route":"{route}","intent":"{intent}","urgency":"{urgency}","confidence":{c}}}'


# ---------------------------------------------------------------- 下游工單
class ImageFinding(BaseModel):
    doc_type: Literal["order_screenshot", "error_screenshot", "invoice", "login_screen",
                      "product_page", "payment_screen", "damage_photo", "unknown"] = "unknown"
    ocr_text: str = ""
    order_ids: list[str] = []
    error_codes: list[str] = []
    quality_ok: bool = True
    quality_note: str = ""


class HandoffTicket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    department: str
    route: str
    intent: str
    urgency: Literal["low", "normal", "high"]
    confidence: float
    order_id: Optional[str] = None
    product: Optional[str] = None
    image_findings: list[ImageFinding] = []
    missing_slots: list[str] = []
    next_action: Literal["handoff", "ask_followup", "refuse", "escalate_human"]
    conversation_summary: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @field_validator("order_id")
    @classmethod
    def _oid(cls, v):
        if v is not None and not ORDER_ID_RE.fullmatch(v):
            raise ValueError("order_id 格式錯誤")
        return v


if __name__ == "__main__":
    print(json.dumps(triage_json_schema(), ensure_ascii=False, indent=2)[:600])
    print(len(ROUTES), "routes;", len(INTENTS), "intents")
