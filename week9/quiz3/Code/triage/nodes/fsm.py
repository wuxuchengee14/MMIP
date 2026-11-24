"""實驗 C：Constrained Decoding 的有限狀態機（FSM / DFA）。

把輸出語言
    {"route":"<R>","intent":"<I of R>","urgency":"<low|normal|high>","confidence":<0.00~0.99|1.0>}
編成 DFA：每輸出一個字元就更新狀態；每個狀態只允許「合法的下一個字元」，其餘 logits 設 -inf（Masking）。
route 與 intent 的對應關係（intent 必須屬於該 route）也寫進狀態，所以連「語意一致性」都由結構保證。

two-level API：
  * 字元級：allowed_chars(state)            —— 給本機字元級 GRU 使用
  * Token 級：token_index(token_strings)     —— 給真實 LLM 的 BPE 詞表使用（一個 token 可能含多個字元）
"""
from __future__ import annotations

from ..spec import ROUTE_INTENTS, URGENCIES

EOS = "\x03"


def _language() -> list[str]:
    confs = [f"0.{i:02d}" for i in range(100)] + ["1.0"]
    out = []
    for r, intents in ROUTE_INTENTS.items():
        for i in intents:
            for u in URGENCIES:
                for c in confs:
                    out.append(f'{{"route":"{r}","intent":"{i}","urgency":"{u}","confidence":{c}}}')
    return out


class JsonFSM:
    def __init__(self):
        # 1) 先建 trie  2) 後序合併相同子樹 → 最小化 DFA（狀態數遠小於 trie）
        trie: list[dict] = [{}]
        acc_trie = set()
        for s in _language():
            n = 0
            for ch in s:
                nxt = trie[n].get(ch)
                if nxt is None:
                    trie.append({})
                    nxt = trie[n][ch] = len(trie) - 1
                n = nxt
            acc_trie.add(n)
        self.trie_nodes = len(trie)
        canon: dict[tuple, int] = {}
        mapping: dict[int, int] = {}
        self.trans: list[dict[str, int]] = []
        self.accept: set[int] = set()

        def build(n: int) -> int:
            if n in mapping:
                return mapping[n]
            key = (n in acc_trie, tuple(sorted((ch, build(c)) for ch, c in trie[n].items())))
            if key not in canon:
                canon[key] = len(self.trans)
                self.trans.append({ch: c for ch, c in key[1]})
                if key[0]:
                    self.accept.add(canon[key])
            mapping[n] = canon[key]
            return mapping[n]

        import sys
        sys.setrecursionlimit(10000)
        self.start = build(0)
        self.n_states = len(self.trans)
        self._tok_cache: dict[int, list[int]] = {}
        self._tok_strs: list[str] | None = None

    # ---- 字元級
    def step(self, s: int, ch: str):
        return self.trans[s].get(ch)

    def walk(self, s: int, text: str):
        for ch in text:
            s = self.trans[s].get(ch)
            if s is None:
                return None
        return s

    def allowed_chars(self, s: int) -> set[str]:
        a = set(self.trans[s])
        if s in self.accept:
            a.add(EOS)
        return a

    def allowed_fn(self):
        """給 CondGRU.generate 使用：prefix 字串 → 合法下一字元集合。"""
        def fn(prefix: str) -> set[str]:
            s = self.walk(self.start, prefix)
            return self.allowed_chars(s) if s is not None else {EOS}
        return fn

    # ---- Token 級（對應 outlines / xgrammar 的作法）
    def token_index(self, token_strings: list[str], eos_id: int | None = None):
        """token_strings[i] 是第 i 個 token 對應的字串。回傳函式 state->合法 token id 列表（含 EOS 於 accept 狀態）。"""
        self._tok_strs = token_strings

        def allowed_tokens(s: int) -> list[int]:
            if s not in self._tok_cache:
                ok = [i for i, t in enumerate(token_strings) if t and self.walk(s, t) is not None]
                if s in self.accept and eos_id is not None:
                    ok.append(eos_id)
                self._tok_cache[s] = ok
            return self._tok_cache[s]
        return allowed_tokens


if __name__ == "__main__":
    f = JsonFSM()
    print("trie nodes", f.trie_nodes, "-> DFA states", f.n_states)
