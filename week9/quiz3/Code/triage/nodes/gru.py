"""從零用 numpy 寫的「條件式字元級 GRU 生成模型」—— 本機沒有 GPU / PyTorch 時，用來實際跑 SFT 與 DPO 的小模型。

它不是 LLM，但訓練流程與損失函數與 LLM 版一致：
  * SFT ：teacher forcing 最大化 log p(target JSON | 使用者文字)
  * DPO ：-log σ( β[(logπ(chosen)-logπref(chosen)) - (logπ(rejected)-logπref(rejected))] )
結構：文字 → TF-IDF(字元 1~3gram, 稀疏) → tanh 線性層得到條件向量 c；c 作為 GRU 初始狀態並在每一步與字元 embedding 串接。
"""
from __future__ import annotations

import pickle

import numpy as np
from scipy import sparse

BOS, EOS = "\x02", "\x03"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class CondGRU:
    def __init__(self, vocab: list[str], in_dim: int, H: int = 128, E: int = 32, seed: int = 0):
        self.vocab = vocab
        self.c2i = {c: i for i, c in enumerate(vocab)}
        self.V, self.H, self.E, self.D = len(vocab), H, E, in_dim
        r = np.random.RandomState(seed)
        g = lambda *s: (r.randn(*s) * (1.0 / np.sqrt(s[0]))).astype(np.float64)
        self.p = {"Wc": g(in_dim, H), "bc": np.zeros(H), "Emb": r.randn(self.V, E) * 0.1,
                  "Wx": g(E + H, 3 * H), "Wh": g(H, 3 * H), "b": np.zeros(3 * H),
                  "Wo": g(H, self.V), "bo": np.zeros(self.V)}

    # ------------------------------------------------------------ 編碼
    def encode_y(self, s: str) -> list[int]:
        return [self.c2i[BOS]] + [self.c2i[c] for c in s] + [self.c2i[EOS]]

    def pad(self, seqs: list[str]):
        ids = [self.encode_y(s) for s in seqs]
        L = max(map(len, ids))
        Y = np.full((len(ids), L), self.c2i[EOS], dtype=np.int64)
        M = np.zeros((len(ids), L - 1))
        for i, a in enumerate(ids):
            Y[i, : len(a)] = a
            M[i, : len(a) - 1] = 1
        return Y, M

    # ------------------------------------------------------------ 前向（可選擇保留 cache 以便反傳）
    def _cond(self, X):
        return np.tanh(X @ self.p["Wc"] + self.p["bc"])

    def _step(self, c, y_prev, h):
        p = self.p
        inp = np.concatenate([p["Emb"][y_prev], c], axis=1)
        gx = inp @ p["Wx"] + p["b"]
        gh = h @ p["Wh"]
        H = self.H
        z = sigmoid(gx[:, :H] + gh[:, :H])
        r = sigmoid(gx[:, H:2 * H] + gh[:, H:2 * H])
        n = np.tanh(gx[:, 2 * H:] + r * gh[:, 2 * H:])
        h2 = (1 - z) * n + z * h
        return h2, (inp, h, z, r, n, gh[:, 2 * H:])

    def forward(self, X, Y, keep=False):
        c = self._cond(X)
        h = c
        B, L = Y.shape
        logits_all, caches = [], []
        for t in range(L - 1):
            h, cache = self._step(c, Y[:, t], h)
            logits_all.append(h @ self.p["Wo"] + self.p["bo"])
            if keep:
                caches.append((cache, h))
        return c, np.stack(logits_all, 1), caches  # logits: (B, L-1, V)

    @staticmethod
    def _logsoftmax(z):
        z = z - z.max(-1, keepdims=True)
        return z - np.log(np.exp(z).sum(-1, keepdims=True))

    def seq_logp(self, X, Y, M):
        _, logits, _ = self.forward(X, Y)
        lp = self._logsoftmax(logits)
        tgt = Y[:, 1:]
        tok = np.take_along_axis(lp, tgt[..., None], -1)[..., 0] * M
        return tok.sum(1)

    # ------------------------------------------------------------ d( Σ_b coef_b · logp_b ) / dθ
    def grad_logp(self, X, Y, M, coef):
        p = self.p
        c, logits, caches = self.forward(X, Y, keep=True)
        lp = self._logsoftmax(logits)
        tgt = Y[:, 1:]
        tok = np.take_along_axis(lp, tgt[..., None], -1)[..., 0] * M
        logp = tok.sum(1)
        probs = np.exp(lp)
        dlog = -probs
        np.put_along_axis(dlog, tgt[..., None], 1 - np.take_along_axis(probs, tgt[..., None], -1), -1)
        dlog *= (M * coef[:, None])[..., None]
        g = {k: np.zeros_like(v) for k, v in p.items()}
        dc = np.zeros_like(c)
        dh_next = np.zeros_like(c)
        H = self.H
        for t in reversed(range(Y.shape[1] - 1)):
            (inp, h_prev, z, r, n, ghn), h_t = caches[t]
            dl = dlog[:, t]
            g["Wo"] += h_t.T @ dl
            g["bo"] += dl.sum(0)
            dh = dl @ p["Wo"].T + dh_next
            dn = dh * (1 - z)
            dz = dh * (h_prev - n)
            dh_prev = dh * z
            dn_pre = dn * (1 - n * n)
            dz_pre = dz * z * (1 - z)
            dr_pre = dn_pre * ghn * r * (1 - r)
            dgx = np.concatenate([dz_pre, dr_pre, dn_pre], 1)
            dgh = np.concatenate([dz_pre, dr_pre, dn_pre * r], 1)
            g["Wx"] += inp.T @ dgx
            g["b"] += dgx.sum(0)
            g["Wh"] += h_prev.T @ dgh
            dinp = dgx @ p["Wx"].T
            dh_prev += dgh @ p["Wh"].T
            np.add.at(g["Emb"], Y[:, t], dinp[:, : self.E])
            dc += dinp[:, self.E:]
            dh_next = dh_prev
        dc += dh_next  # h0 = c
        dpre = dc * (1 - c * c)
        g["Wc"] = np.asarray(X.T @ dpre) if sparse.issparse(X) else X.T @ dpre
        g["bc"] = dpre.sum(0)
        return logp, g

    # ------------------------------------------------------------ 生成
    def generate(self, x_row, allowed_fn=None, temperature: float = 0.0, rng=None, max_len: int = 130):
        """allowed_fn(prefix:str)->set[str]|None：Constrained Decoding 的 mask 來源；None 表示不限制。
        回傳 (字串, 實際用到的步數)。"""
        c = self._cond(x_row)
        h = c
        y = np.array([self.c2i[BOS]])
        out = []
        for _ in range(max_len):
            h, _ = self._step(c, y, h)
            logits = (h @ self.p["Wo"] + self.p["bo"])[0]
            if allowed_fn is not None:
                allow = allowed_fn("".join(out))
                mask = np.full(self.V, -1e30)
                for ch in allow:
                    mask[self.c2i[ch]] = 0.0
                logits = logits + mask
            if temperature <= 0:
                k = int(np.argmax(logits))
            else:
                pr = np.exp((logits - logits.max()) / temperature)
                pr /= pr.sum()
                k = int(rng.choice(self.V, p=pr))
            ch = self.vocab[k]
            if ch == EOS:
                break
            out.append(ch)
            y = np.array([k])
        return "".join(out)

    def clone_params(self):
        return {k: v.copy() for k, v in self.p.items()}

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"vocab": self.vocab, "p": self.p, "H": self.H, "E": self.E, "D": self.D}, f)

    @staticmethod
    def load(path):
        d = pickle.load(open(path, "rb"))
        m = CondGRU(d["vocab"], d["D"], d["H"], d["E"])
        m.p = d["p"]
        return m


class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, clip=5.0):
        self.lr, self.b1, self.b2, self.clip, self.t = lr, b1, b2, clip, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        """grads 是「目標函數的上升方向」；這裡做梯度下降於 -目標。"""
        self.t += 1
        norm = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
        s = min(1.0, self.clip / (norm + 1e-9))
        for k in params:
            g = -grads[k] * s
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)
        return norm
