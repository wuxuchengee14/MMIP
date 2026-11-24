"""把多輪對話 + 圖片 OCR 組成分流節點的輸入文字（A/B/C/D 共用，訓練時也用同一格式）。"""
from __future__ import annotations

MAX_TURNS = 3  # 只看最近 3 則使用者訊息，控制長度與干擾


def build_input(user_turns: list[str], image_texts: list[str] | None = None) -> str:
    turns = [t.strip() for t in user_turns if t and t.strip()][-MAX_TURNS:]
    text = " / ".join(turns)
    if image_texts:
        ocr = " ".join(x for x in image_texts if x)
        if ocr:
            text = f"{text} [圖片] {ocr}".strip()
    return text
