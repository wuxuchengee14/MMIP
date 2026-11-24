"""實驗 B（SFT / DPO）與實驗 C（Constrained Decoding）共用的 GRU 節點與訓練程式。"""
from __future__ import annotations

import pickle
import random
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..data import corrupt, target_json
from .base import Node
from .fsm import JsonFSM
from .gru import BOS, EOS, Adam, CondGRU

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"


def build_vocab(rows) -> list[str]:
    rng = random.Random(0)
    chars = {BOS, EOS}
    for r in rows:
        js = target_json(r, rng)
        chars |= set(js)
        for _ in range(4):
            chars |= set(corrupt(js, rng))
    return sorted(chars)


def make_vec(rows) -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True, min_df=2,
                           max_features=4000).fit([r["text"] for r in rows])


def train_sft(rows, vec, vocab, epochs=30, bs=32, lr=3e-3, H=128, seed=0, log=print) -> tuple[CondGRU, list]:
    rng = random.Random(seed)
    m = CondGRU(vocab, in_dim=len(vec.vocabulary_), H=H, seed=seed)
    opt = Adam(m.p, lr=lr)
    tg = random.Random(1)
    X_all = vec.transform([r["text"] for r in rows])
    Y_all = [target_json(r, tg) for r in rows]
    idx = list(range(len(rows)))
    hist = []
    for ep in range(epochs):
        rng.shuffle(idx)
        tot, ntok = 0.0, 0
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            Y, M = m.pad([Y_all[j] for j in b])
            coef = np.full(len(b), 1.0 / len(b))
            lp, g = m.grad_logp(X_all[b], Y, M, coef)
            opt.step(m.p, g)
            tot += -lp.sum(); ntok += M.sum()
        hist.append(tot / ntok)
        if ep % 5 == 0 or ep == epochs - 1:
            log(f"  [SFT] epoch {ep+1:02d}/{epochs} nll/token={tot/ntok:.4f}")
    return m, hist


def train_dpo(ref: CondGRU, rows, vec, epochs=4, bs=32, lr=1e-3, beta=0.5, sft_alpha=0.2, seed=0, log=print):
    """DPO（再加一個小比例的 SFT 錨定項，避免 chosen 機率也被一起壓低，即 RPO 的做法）。"""
    rng = random.Random(seed)
    m = CondGRU(ref.vocab, ref.D, ref.H, ref.E)
    m.p = ref.clone_params()
    opt = Adam(m.p, lr=lr)
    tg = random.Random(2)
    X_all = vec.transform([r["text"] for r in rows])
    ch = [target_json(r, tg) for r in rows]
    rj = [corrupt(c, tg) for c in ch]
    idx = list(range(len(rows)))
    hist = []
    for ep in range(epochs):
        rng.shuffle(idx)
        accs, losses = [], []
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            Yc, Mc = m.pad([ch[j] for j in b]); Yr, Mr = m.pad([rj[j] for j in b])
            X = X_all[b]
            ref_c, ref_r = ref.seq_logp(X, Yc, Mc), ref.seq_logp(X, Yr, Mr)
            pc, pr = m.seq_logp(X, Yc, Mc), m.seq_logp(X, Yr, Mr)
            z = beta * ((pc - ref_c) - (pr - ref_r))
            w = 1 / (1 + np.exp(z))                       # σ(-z)
            n = len(b)
            _, gc = m.grad_logp(X, Yc, Mc, w * beta / n + sft_alpha / n)
            _, gr = m.grad_logp(X, Yr, Mr, -w * beta / n)
            g = {k: gc[k] + gr[k] for k in gc}
            opt.step(m.p, g)
            losses.append(float(np.mean(np.log1p(np.exp(-z))))); accs.append(float((z > 0).mean()))
        hist.append((np.mean(losses), np.mean(accs)))
        log(f"  [DPO] epoch {ep+1}/{epochs} loss={np.mean(losses):.4f} pref-acc={np.mean(accs):.3f}")
    return m, hist


class GRUNode(Node):
    """temperature=0 為 greedy；>0 為隨機取樣。constrained=True 時套用 FSM mask（實驗 C）。"""

    def __init__(self, model: CondGRU, vec, name: str, temperature=0.0, constrained=False, seed=0, fsm: JsonFSM | None = None):
        self.model, self.vec, self.name = model, vec, name
        self.temperature, self.constrained = temperature, constrained
        self.rng = np.random.RandomState(seed)
        self.fsm = fsm or (JsonFSM() if constrained else None)
        self._allow = self.fsm.allowed_fn() if constrained else None

    def _run(self, text):
        x = self.vec.transform([text])
        s = self.model.generate(x, allowed_fn=self._allow, temperature=self.temperature, rng=self.rng)
        return s, {}


def save_bundle(path, **objs):
    with open(path, "wb") as f:
        pickle.dump(objs, f)


def load_bundle(path):
    return pickle.load(open(path, "rb"))
