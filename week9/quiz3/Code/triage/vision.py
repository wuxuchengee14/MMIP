"""圖片感知層。

介面（analyze_image → ImageFinding）與「VLM 版」相同；本機版以 OCR + 影像品質檢查實作：
  - 品質檢查：Laplacian 變異數（模糊）、平均亮度（過暗/過曝）、解析度
  - OCR：tesseract（繁中 + 英文）
  - 文件類型：依 OCR 關鍵字判斷
  - 實體抽取：訂單編號、錯誤碼
正式環境可把 analyze_image 換成 Qwen2.5-VL 等 VLM（輸出同一個 ImageFinding schema）。
注意：純視覺線索（例如「外殼有裂痕」但圖上沒有文字）本機版看不出來，這是已知限制。
"""
from __future__ import annotations

import re
import os
import shutil
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from .spec import ORDER_ID_RE, ImageFinding

BLUR_THRESHOLD = 60.0      # Laplacian variance 低於此值視為模糊
DARK_THRESHOLD = 45.0      # 平均亮度低於此值視為過暗
BRIGHT_THRESHOLD = 235.0
MIN_SIDE = 200

DOC_KEYWORDS = [
    ("payment_screen", ["付款失敗", "已扣款", "交易編號", "payment failed"]),
    ("login_screen", ["登入失敗", "帳號或密碼", "login failed", "忘記密碼"]),
    ("invoice", ["發票", "統編", "統一編號"]),
    ("error_screenshot", ["錯誤碼", "錯誤代碼", "error", "無法連線"]),
    ("damage_photo", ["破損", "damaged", "瑕疵", "裂痕"]),
    ("order_screenshot", ["訂單編號", "訂單詳情", "物流", "已簽收", "鑑賞期"]),
    ("product_page", ["規格", "pcie", "建議電源", "gb"]),
]
ERR_RE = re.compile(r"(?:E-?\d{3}|Error\s*\d{3}|code\s*\d{3}|錯誤碼\s*[A-Z]?-?\d{3})", re.I)


@lru_cache(maxsize=1)
def configure_ocr():
    workspace_runtime = Path(__file__).resolve().parents[3] / '.runtime' / 'tesseract'
    program_files = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Tesseract-OCR'
    options = [os.environ.get('WEEK9_TESSERACT'), shutil.which('tesseract'),
               workspace_runtime / 'tesseract.exe', program_files / 'tesseract.exe',
               workspace_runtime.parent / 'ocr' / 'Library' / 'bin' / 'tesseract.exe']
    executable = next((str(path) for path in options if path and Path(path).is_file()), None)
    if executable is None:
        raise RuntimeError('Tesseract executable unavailable')
    pytesseract.pytesseract.tesseract_cmd = executable
    data_options = [Path(executable).parent / 'tessdata', workspace_runtime.parent / 'ocr' / 'share' / 'tessdata']
    for data in data_options:
        if (data / 'chi_tra.traineddata').is_file() and (data / 'eng.traineddata').is_file():
            os.environ.setdefault('TESSDATA_PREFIX', str(data))
            break
    if not {'chi_tra', 'eng'}.issubset(pytesseract.get_languages()):
        raise RuntimeError('Tesseract chi_tra and eng language data are required')
    return executable


def _decode(data) -> np.ndarray:
    if isinstance(data, (bytes, bytearray)):
        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.imread(str(data), cv2.IMREAD_COLOR)


def quality_check(img: np.ndarray) -> tuple[bool, str]:
    if img is None:
        return False, "無法讀取圖片"
    h, w = img.shape[:2]
    if min(h, w) < MIN_SIDE:
        return False, "解析度太低"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean = float(gray.mean())
    if mean < DARK_THRESHOLD:
        return False, "太暗"
    if mean > BRIGHT_THRESHOLD:
        return False, "過曝"
    if lap < BLUR_THRESHOLD:
        return False, "太模糊"
    return True, ""


def ocr(img: np.ndarray) -> str:
    configure_ocr()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if min(gray.shape) < 700:  # 放大小圖提升 OCR
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    txt = pytesseract.image_to_string(gray, lang="chi_tra+eng", config="--psm 6", timeout=10)
    # tesseract 對中文會在字間插空白；移除「中文與中文之間」的空白
    txt = re.sub(r"(?<=[一-鿿])[ \t]+(?=[一-鿿])", "", txt)
    return re.sub(r"[ \t]+", " ", txt).strip()


def analyze_image(data) -> ImageFinding:
    img = _decode(data)
    ok, note = quality_check(img)
    if not ok:
        return ImageFinding(quality_ok=False, quality_note=note)
    try:
        text = ocr(img)
    except (RuntimeError, OSError):
        return ImageFinding(quality_ok=False, quality_note="目前無法辨識")
    low = text.lower()
    doc = "unknown"
    for name, kws in DOC_KEYWORDS:
        if any(k in low for k in kws):
            doc = name
            break
    return ImageFinding(
        doc_type=doc, ocr_text=" ".join(text.split()),
        order_ids=sorted(set(ORDER_ID_RE.findall(text))),
        error_codes=sorted({m.group(0) for m in ERR_RE.finditer(text)}),
        quality_ok=True, quality_note="")
