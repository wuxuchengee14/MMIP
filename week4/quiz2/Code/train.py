from __future__ import annotations

import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from common.metrics import classification_metrics
from common.training import amp_context, fit
from common.utils import (ROOT, configure, digest, loader, read_json, report, seed_all,
                          synchronize, table, write_csv, write_json)
from quiz2.Code.models import SentimentModel


def step(model, batch, device):
    tokens, lengths, labels = [x.to(device, non_blocking=True) for x in batch]
    logits = model(tokens, lengths)
    return F.cross_entropy(logits, labels), len(labels), int((logits.argmax(-1) == labels).sum())


def run(config, device):
    settings = config["text"]
    folder = ROOT / "quiz1" / "Data" / "imdb"
    manifest = read_json(folder / "manifest.json")
    vocab = read_json(folder / "vocab.json")
    datasets, records = {}, {}
    for split in ("train", "val", "test"):
        data = torch.load(folder / f"{split}.pt", weights_only=True)
        records[split] = data
        datasets[split] = TensorDataset(data["tokens"], data["lengths"], data["labels"])
    loaders = {s: loader(d, config, settings["batch_size"], shuffle=s == "train") for s, d in datasets.items()}
    comparison = []
    for kind in ("rnn", "lstm"):
        seed_all(config["seed"])
        model = SentimentModel(kind, len(vocab), settings["embedding_dim"], settings["hidden_dim"],
                               settings["dropout"]).to(device)
        directory = ROOT / "quiz2" / "Output" / kind
        done = fit(model, loaders["train"], loaders["val"], step, settings, directory, device,
                   {"model": kind, "seed": config["seed"], "data": digest(manifest)})
        probs = []
        model.eval()
        # Warm up CUDA separately; total test time also includes data transfer/loading.
        first = next(iter(loaders["test"]))
        with torch.inference_mode(), amp_context(device):
            for _ in range(3):
                model(first[0].to(device), first[1].to(device))
        synchronize(device)
        tick = time.perf_counter()
        with torch.inference_mode(), amp_context(device):
            for tokens, lengths, _ in loaders["test"]:
                probs.append(model(tokens.to(device, non_blocking=True), lengths.to(device, non_blocking=True))
                             .float().softmax(-1).cpu().numpy())
        synchronize(device)
        seconds = time.perf_counter() - tick
        metrics = classification_metrics(records["test"]["labels"].numpy(), np.concatenate(probs),
                                         ["negative", "positive"], directory, records["test"]["ids"])
        # Preserve readable review snippets for a balanced success/error case study.
        p = np.concatenate(probs)
        examples = []
        for correct in (True, False):
            ids = np.flatnonzero((p.argmax(1) == records["test"]["labels"].numpy()) == correct)[:8]
            for i in ids:
                sid = records["test"]["ids"][i]
                path = folder / "raw" / "aclImdb" / sid
                examples.append({"id": sid, "correct": correct, "label": int(records["test"]["labels"][i]),
                                 "prediction": int(p[i].argmax()), "text": path.read_text(encoding="utf-8")[:1500]})
        write_json(directory / "examples.json", examples)
        timing = {"end_to_end_test_seconds": seconds, "reviews_per_second": len(p) / seconds,
                  "batch_size": settings["batch_size"], "includes": "loading + host transfer + forward + CPU probabilities"}
        write_json(directory / "timing.json", timing)
        comparison.append({"model": kind, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                           "roc_auc": metrics["positive_auc"], "parameters": done["parameters"],
                           "training_seconds": done["training_seconds"], "test_seconds": seconds,
                           "best_epoch": done["best_epoch"]})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_csv(ROOT / "quiz2" / "Output" / "comparison.csv", comparison)
    winner = max(comparison, key=lambda r: r["macro_f1"])
    report("quiz2", "Quiz 2：RNN 與 LSTM 情感分析",
           "## 方法\n\nIMDb 正負二分類；單向 RNN / LSTM 共用相同詞彙表、Embedding 維度、"
           "hidden 維度、訓練／驗證／測試分割和 early stopping。取最後一個有效 token 的狀態分類，"
           "避免 padding 污染；截斷至前 256 tokens（實際長度依設定檔）。"
           "交叉熵、AdamW、梯度裁切，驗證 Accuracy 選擇權重，測試不參與挑選。\n\n"
           "## 測試比較\n\n" + table(comparison) +
           f"\n\n本次測試 Macro-F1 較高者為 {winner['model']}。"
           "單次固定 seed 的結果不代表統計顯著差異；RNN 可能因長序列記憶不足而落後，"
           "應搭配 examples.json 的實際錯誤閱讀，不先假定 LSTM 必定勝出。\n\n"
           "## 檔案\n\n各模型子目錄包含 best.pt、last.pt、history.csv、learning_curves.png、"
           "metrics.json、classification_report.json、confusion_matrix.png、roc.png、"
           "predictions.csv、examples.json 與 timing.json。Top-5 在二分類等於 1，"
           "因此本題比較以 Accuracy、F1、正類 ROC-AUC 為主。\n\n"
           "設定與資料識別見 complete.json。時間是實際測量，測試時間包含載入與傳輸。")
    return comparison


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.rtx5090.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = read_json(args.config)
    run(config, configure(config, args.device))
