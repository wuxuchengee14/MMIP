"""實驗 D：System One 分流模型 —— 一個小而快的分類器，只做「分流」這一件事（類 Clef / Cloudflare decision model 的思路）。
字元 1~3-gram TF-IDF + Logistic Regression；confidence 來自 predict_proba。
輸出仍組成與 LLM 相同的 JSON 字串，讓四種節點可以用同一套評估器比較。"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ..paths import ROUTING_DATA
from ..spec import INTENT2ROUTE, canonical
from .base import Node

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "results" / "system_one.pkl"


class SystemOne(Node):
    name = "D_system_one"

    def __init__(self, vec=None, clf_intent=None, clf_urg=None):
        self.vec, self.clf_intent, self.clf_urg = vec, clf_intent, clf_urg

    @staticmethod
    def train(rows: list[dict], C: float = 20.0) -> "SystemOne":
        X_text = [r["text"] for r in rows]
        vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True, min_df=2)
        X = vec.fit_transform(X_text)
        ci = LogisticRegression(C=C, max_iter=3000).fit(X, [r["intent"] for r in rows])
        cu = LogisticRegression(C=C, max_iter=3000).fit(X, [r["urgency"] for r in rows])
        return SystemOne(vec, ci, cu)

    def proba(self, text: str):
        x = self.vec.transform([text])
        return self.clf_intent.predict_proba(x)[0], self.clf_urg.predict_proba(x)[0]

    def predict(self, text: str) -> dict:
        pi, pu = self.proba(text)
        k = int(np.argmax(pi))
        intent = str(self.clf_intent.classes_[k])
        return {"intent": intent, "route": INTENT2ROUTE[intent], "urgency": str(self.clf_urg.classes_[int(np.argmax(pu))]),
                "confidence": float(pi[k]), "p_intent": {str(c): float(v) for c, v in zip(self.clf_intent.classes_, pi)}}

    def _run(self, text):
        p = self.predict(text)
        return canonical(p["route"], p["intent"], p["urgency"], min(p["confidence"], 0.99)), {"confidence": p["confidence"]}

    def save(self, path=MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump((self.vec, self.clf_intent, self.clf_urg), f)

    @staticmethod
    def load(path=MODEL_PATH) -> "SystemOne":
        with open(path, "rb") as f:
            return SystemOne(*pickle.load(f))


def load_train_rows():
    return json.loads((ROUTING_DATA / "train_rows.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    m = SystemOne.train(load_train_rows()); m.save()
    from ..spec import validate_triage
    n = ok = 0
    for l in open(ROUTING_DATA / "benchmark.jsonl", encoding="utf-8"):
        c = json.loads(l); r = m.run(c["input"]); o, _ = validate_triage(r.raw)
        n += 1; ok += int(o is not None and o.intent == c["gold"]["intent"])
        if not (o and o.intent == c["gold"]["intent"]):
            print(c["id"], c["input"][:50], "->", o.intent if o else r.raw, "| gold", c["gold"]["intent"])
    print(f"intent acc {ok}/{n}")
