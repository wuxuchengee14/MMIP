"""訓練實驗 B 的三個模型：B0 少量資料 SFT、B1 完整 SFT、B2 = B1 + DPO。"""
import json, random, time
from pathlib import Path
from triage.nodes.sft_node import *
ROOT = Path(__file__).resolve().parents[1]
from triage.nodes.sft_node import RES
from triage.nodes.system_one import load_train_rows

RES.mkdir(parents=True, exist_ok=True)

rows = load_train_rows()
random.Random(3).shuffle(rows)
val, train = rows[:86], rows[86:]                    # 10% 當驗證
vec = make_vec(train); vocab = build_vocab(rows)
log_lines = []
def log(s): print(s, flush=True); log_lines.append(s)

# B0：每個 intent 只給 8 筆（資料效率對照）
small, cnt = [], {}
for r in train:
    if cnt.get(r["intent"], 0) < 8:
        small.append(r); cnt[r["intent"]] = cnt.get(r["intent"], 0) + 1
log(f"train={len(train)} val={len(val)} small={len(small)} vocab={len(vocab)} dim={len(vec.vocabulary_)}")
t0 = time.time(); b0, h0 = train_sft(small, vec, vocab, epochs=30, log=log); log(f"B0 train time {time.time()-t0:.1f}s")
t0 = time.time(); b1, h1 = train_sft(train, vec, vocab, epochs=30, log=log); t_sft = time.time()-t0; log(f"B1 train time {t_sft:.1f}s")
t0 = time.time(); b2, h2 = train_dpo(b1, train, vec, epochs=4, log=log); t_dpo = time.time()-t0; log(f"B2 DPO time {t_dpo:.1f}s")
save_bundle(RES / "bundle_b.pkl", vec=vec, vocab=vocab, B0=b0.p, B1=b1.p, B2=b2.p, hist={"B0": h0, "B1": h1, "B2": h2},
            val=val, dims=(b1.D, b1.H, b1.E), times={"B1_sft_s": t_sft, "B2_dpo_s": t_dpo})
(RES / "train_b.log").write_text("\n".join(log_lines), encoding="utf-8")
