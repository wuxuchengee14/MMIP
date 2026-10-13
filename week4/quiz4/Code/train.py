from __future__ import annotations

import sys
import time
import textwrap
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from common.training import amp_context, fit
from common.utils import (ROOT, configure, digest, loader, read_json, report, seed_all,
                          synchronize, table, write_csv, write_json)
from quiz1.Code.prepare import encode
from quiz4.Code.bleu import corpus_bleu
from quiz4.Code.features import ImageDataset, encoder, feature_tokens, prepare_features
from quiz4.Code.models import CaptionModel, decode, generate


class CaptionDataset(Dataset):
    def __init__(self, features, records, vocab, max_length):
        self.features = features
        words = {word: i for i, word in enumerate(vocab)}
        self.image_indices, captions = [], []
        for i, record in enumerate(records):
            for text in record["captions"]:
                ids = encode(text, words, max_length, caption=True)
                captions.append(ids + [0] * (max_length - len(ids)))
                self.image_indices.append(i)
        self.tokens = torch.tensor(captions, dtype=torch.long)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return self.features[self.image_indices[index]].float(), self.tokens[index]


def step(model, batch, device):
    features, tokens = [x.to(device, non_blocking=True) for x in batch]
    logits = model(features, tokens[:, :-1])
    targets = tokens[:, 1:]
    valid = targets.ne(0)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0)
    return loss, int(valid.sum()), int(((logits.argmax(-1) == targets) & valid).sum())


def predict(model, features, vocab, settings, device, width):
    outputs = []
    model.eval()
    with torch.inference_mode(), amp_context(device):
        for start in range(0, len(features), settings["decode_batch_size"]):
            batch = features[start:start + settings["decode_batch_size"]].to(device).float()
            ids = generate(model, batch, settings["max_length"], width, settings["beam_length_penalty"])
            outputs.extend(decode(row, vocab) for row in ids.cpu().tolist())
    return outputs


def benchmark(model, image_features, settings, device, width):
    # Batch=1 latency, same image features/repeats for every method.
    n = min(settings["timing_samples"], len(image_features))
    with torch.inference_mode(), amp_context(device):
        first = image_features[:1].float().to(device)
        for _ in range(3):
            generate(model, first, settings["max_length"], width, settings["beam_length_penalty"])
        times = []
        token_counts = []
        for _ in range(settings["timing_repeats"]):
            for i in range(n):
                feat = image_features[i:i+1].float().to(device)
                synchronize(device)
                tick = time.perf_counter()
                ids = generate(model, feat, settings["max_length"], width, settings["beam_length_penalty"])
                synchronize(device)
                times.append(time.perf_counter() - tick)
                token_counts.append(sum(t not in (0, 2, 3) for t in ids[0].tolist()))
    return {"decoder_mean_ms": float(np.mean(times) * 1000),
            "decoder_median_ms": float(np.median(times) * 1000),
            "decoder_p95_ms": float(np.percentile(times, 95) * 1000),
            "mean_generated_tokens": float(np.mean(token_counts)),
            "generated_tokens_per_second": sum(token_counts) / sum(times),
            "timing_images": n, "timing_repeats": settings["timing_repeats"], "latency_batch_size": 1}


def benchmark_encoder(records, settings, device):
    folder = ROOT / "quiz1" / "Data" / "flickr8k"
    images = ImageDataset(folder, records)
    net = encoder(settings["grid_size"]).to(device)
    n = min(settings["timing_samples"], len(images))
    times = []
    with torch.inference_mode(), amp_context(device):
        first = images[0][None].to(device)
        for _ in range(3):
            feature_tokens(net, first)
        for _ in range(settings["timing_repeats"]):
            for i in range(n):
                synchronize(device)
                tick = time.perf_counter()
                # Includes image open, deterministic transforms and host transfer.
                image = images[i][None].to(device)
                feature_tokens(net, image)
                synchronize(device)
                times.append(time.perf_counter() - tick)
    del net
    return {"encoder_mean_ms": float(np.mean(times) * 1000),
            "encoder_p95_ms": float(np.percentile(times, 95) * 1000),
            "timing_images": n, "repeats": settings["timing_repeats"],
            "includes": "image read + preprocessing + transfer + ResNet forward"}


def plot_examples(records, outputs, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    folder = ROOT / "quiz1" / "Data" / "flickr8k"
    fig, axes = plt.subplots(3, 2, figsize=(13, 15))
    for ax, record, text in zip(axes.ravel(), records[:6], outputs[:6]):
        with Image.open(folder / record["image"]) as image:
            ax.imshow(image)
        ax.axis("off")
        ax.set_title(textwrap.fill("Generated: " + text, 55) + "\n" +
                     textwrap.fill("Reference: " + record["captions"][0], 55), fontsize=9)
    fig.tight_layout()
    fig.savefig(directory / "caption_examples.png", dpi=140)
    plt.close(fig)


def run(config, device):
    settings = config["caption"]
    feature_info = prepare_features(config, device)
    folder = ROOT / "quiz1" / "Data" / "flickr8k"
    vocab = read_json(folder / "vocab.json")
    records = {s: read_json(folder / f"{s}.json") for s in ("train", "val", "test")}
    cached = {s: torch.load(ROOT / "quiz4" / "Data" / "features" / f"{s}.pt", weights_only=True)
              for s in records}
    for s in records:
        if cached[s]["ids"] != [r["id"] for r in records[s]]:
            raise ValueError("Feature/image order mismatch")
    features = {s: cached[s]["features"] for s in records}
    # Feature cache is already RAM-resident; avoid Windows worker copies/IPC.
    feature_config = dict(config, workers=0)
    loaders = {
        s: loader(CaptionDataset(features[s], records[s], vocab, settings["max_length"]),
                  feature_config, settings["batch_size"], shuffle=s == "train")
        for s in ("train", "val")
    }
    encoder_timing = benchmark_encoder(records["test"], settings, device)
    write_json(ROOT / "quiz4" / "Output" / "encoder_timing.json", encoder_timing)
    comparison = []
    for kind in ("lstm", "transformer"):
        seed_all(config["seed"])
        model = CaptionModel(kind, len(vocab), **{k: settings[k] for k in
                             ("dim", "layers", "heads", "dropout", "max_length", "grid_size")}).to(device)
        directory = ROOT / "quiz4" / "Output" / kind
        done = fit(model, loaders["train"], loaders["val"], step, settings, directory, device,
                   {"model": kind, "seed": config["seed"], "features": feature_info["identity"],
                    "vocab": digest(vocab)}, maximize=False)
        for mode, width in (("greedy", 1), ("beam", settings["beam_width"])):
            predictions = predict(model, features["test"], vocab, settings, device, width)
            metrics = corpus_bleu([r["captions"] for r in records["test"]], predictions)
            timing = benchmark(model, features["test"], settings, device, width)
            timing["estimated_pipeline_mean_ms"] = timing["decoder_mean_ms"] + encoder_timing["encoder_mean_ms"]
            timing["note"] = "Pipeline mean is sum of separately measured encoder and decoder means."
            result_dir = directory / mode
            result_dir.mkdir(parents=True, exist_ok=True)
            write_json(result_dir / "bleu.json", metrics)
            write_json(result_dir / "timing.json", timing)
            rows = [{"id": r["id"], "image": r["image"], "caption": text, "references": r["captions"]}
                    for r, text in zip(records["test"], predictions)]
            write_json(result_dir / "predictions.json", rows)
            write_csv(result_dir / "predictions.csv",
                      [{**r, "references": " || ".join(r["references"])} for r in rows])
            plot_examples(records["test"], predictions, result_dir)
            comparison.append({"model": kind, "decoding": mode,
                               **{k: metrics[k] for k in ("bleu1", "bleu2", "bleu3", "bleu4")},
                               "decoder_mean_ms": timing["decoder_mean_ms"],
                               "estimated_pipeline_mean_ms": timing["estimated_pipeline_mean_ms"],
                               "mean_generated_tokens": timing["mean_generated_tokens"],
                               "training_seconds": done["training_seconds"],
                               "decoder_parameters": done["parameters"],
                               "total_parameters": done["parameters"] + feature_info["encoder_parameters"]})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_csv(ROOT / "quiz4" / "Output" / "comparison.csv", comparison)
    write_json(ROOT / "quiz4" / "Output" / "comparison.json", comparison)
    make_report(comparison, feature_info)
    return comparison


def make_report(comparison=None, feature_info=None):
    output = ROOT / "quiz4" / "Output"
    comparison = comparison or read_json(output / "comparison.json")
    feature_file = ROOT / "quiz4" / "Data" / "features" / "complete.json"
    if not feature_file.exists():
        feature_file = output / "feature_extraction_metadata.json"
    feature_info = feature_info or read_json(feature_file)
    gemini = read_json(output / "gemini" / "summary.json") if (output / "gemini" / "summary.json").exists() else None
    winner = max(comparison, key=lambda r: r["bleu4"])
    status = table(gemini["comparison"]) if gemini else "**待完成：尚無實際 Gemini 評估結果。**"
    if gemini and not gemini["complete"]:
        status += "\n\n**Gemini API 方式尚有失敗／缺漏項目。**"
    manual_file = output / "gemini_manual" / "summary.json"
    if manual_file.exists():
        manual = read_json(manual_file)
        status += "\n\n### 手動上傳 Gemini 網頁\n\n" + table(manual.get("comparison", []))
        status += ("\n\n手動評估已完整匯入。" if manual.get("complete") else "\n\n**手動評估尚未完成。**")
        status += (" 與 API 結果分開統計，使用匿名 A/B 描述逐一評分；"
                   "模型/模式名稱由使用者依介面記錄。"
                   "詳見 [手動評估報告](gemini_manual/report.md) 與 [操作說明](gemini_manual/README.md)。")
    elif not gemini or not gemini["complete"]:
        status += "\n\n**Gemini 評估未完成。**"
    cases_file = output / "gemini_cases10" / "summary.json"
    if cases_file.exists():
        cases = read_json(cases_file)
        status += "\n\n### 10 張代表案例展示（手動上傳）\n\n" + table(cases.get("comparison", []))
        status += ("\n\n十張圖片、兩模型共二十筆評估已完整匯入。" if cases.get("complete") else
                   "\n\n**十張案例評估尚未完成。**")
        status += (" 依使用者要求人工挑選不同內容及錯誤類型，屬案例展示，並非隨機抽樣的效能估計。"
                   "與 API 及 100 張手動評估分開統計。"
                   "詳見 [案例評估報告](gemini_cases10/report.md) 與 [選例說明](gemini_cases10/CASES.md)。")
        if cases.get("complete"):
            # Present the selected final submission method, rather than abandoned pending runs.
            label = cases.get("session", {}).get("evaluator_model_label", "未記錄")
            status = ("本次繳交採手動上傳 Gemini 網頁的 10 張代表案例，"
                      "每張獨立評估 LSTM / Transformer 的 greedy 描述，共 20/20 筆，已完整匯入。"
                      "介面顯示模型/模式：" + label + "（使用者記錄，非 API 後端版本）。\n\n" +
                      table(cases["comparison"]) +
                      "\n\n案例由人工依內容與錯誤類型挑選，A/B 描述位置平衡隨機；"
                      "未向評估者提供模型名稱或人工參考句。符合率僅限這十張案例，不能外推全體測試集。"
                      "先前 API 嘗試及未使用的 100 張手動方案不混入此表。"
                      "詳見 [完整案例評估報告](gemini_cases10/report.md)、"
                      "[選例說明](gemini_cases10/CASES.md) 與 responses/ 原始回覆。")
    report("quiz4", "Quiz 4：影像描述模型訓練與比較",
           "## 方法\n\nFlickr8k 官方 image-level 分割，每張圖五段描述。固定 ImageNet "
           "ResNet-50 編碼器，空間特徵快取；LSTM 與具 cross-attention 的 Transformer Decoder "
           "分別從頭訓練。LSTM 以平均影像特徵初始化 hidden/cell；Transformer 使用空間位置編碼。"
           "訓練採 teacher forcing、PAD 忽略交叉熵，依 validation token loss 選擇權重。"
           "測試只輸入圖片特徵與 BOS，逐字生成到 EOS 或最大長度，不輸入參考句。\n\n"
           "## BLEU 與時間比較\n\n" + table(comparison) +
           f"\n\n本次 BLEU-4 最高為 {winner['model']} / {winner['decoding']}。"
           "BLEU 為 corpus multi-reference BLEU-1～4，使用全部測試影像與每張五段參考句，"
           "統一 tokenize、無 smoothing、分數介於 0～1。詞彙表只由訓練描述建立。\n\n"
           "推論時間是相同測試樣本、batch=1、暖機後重複測量且 CUDA 同步。"
           "decoder_mean_ms 使用預先算好的影像特徵；estimated_pipeline_mean_ms 為另行測量的"
           "影像讀取／前處理／編碼平均時間加上 decoder 平均時間，屬估計總和。"
           "句長會影響速度，另記錄生成 token 數及 tokens/sec。Gemini 網路時間分開保存。\n\n"
           f"共用影像特徵擷取耗時 {feature_info['extraction_seconds']:.2f} 秒，"
           "不重複算入兩個 decoder 的訓練時間。\n\n"
           "## Gemini 影像符合度評估\n\n" + status +
           "\n\n兩種模型使用同一批圖片，greedy 描述分別評估；"
           "評估主體、動作、場景、錯誤細節，輸出符合度與理由。"
           "Gemini 是輔助評估者，判斷可能有誤，仍應人工檢查。"
           "summary.json 記錄成功數、預期數、模型版本、prompt 與完成狀態。\n\n"
           "## 檔案\n\n各模型包含 best.pt、last.pt、history.csv、learning_curves.png；"
           "greedy/beam 目錄包含 predictions.json/csv、bleu.json、timing.json、caption_examples.png。"
           "本次手動 Gemini 原始回覆、圖片、A/B 對照與評分位於 gemini_cases10/；"
           "先前 API 嘗試保存在本機 gemini/。")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.rtx5090.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = read_json(args.config)
    run(config, configure(config, args.device))
