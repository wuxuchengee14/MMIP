"""分流節點共同介面：每個節點吃一段文字，吐出「原始輸出字串」（之後才驗證），並回報延遲與成本。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

CPU_USD_PER_SEC = 0.04 / 3600  # 假設：雲端單一 vCPU 每小時約 0.04 美元（實驗成本估算用，可調整）


@dataclass
class NodeResult:
    raw: str                    # 模型原始輸出（尚未驗證）
    latency_ms: float
    cost_usd: float
    extra: dict = field(default_factory=dict)  # 例如 token 數、信心、是否 fallback


class Node:
    name = "base"

    def _run(self, text: str) -> tuple[str, dict]:  # 子類別實作
        raise NotImplementedError

    def run(self, text: str) -> NodeResult:
        t0 = time.perf_counter()
        raw, extra = self._run(text)
        dt = time.perf_counter() - t0
        cost = extra.pop("cost_usd", None)
        return NodeResult(raw, dt * 1000, dt * CPU_USD_PER_SEC if cost is None else cost, extra)
