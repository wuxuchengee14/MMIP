from __future__ import annotations

import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import models

from common.metrics import classification_metrics
from common.training import amp_context, fit
from common.utils import (ROOT, configure, digest, loader, read_json, report, seed_all,
                          synchronize, table, write_csv, write_json)

WEIGHTS = {
    "vit_b_16": models.ViT_B_16_Weights.IMAGENET1K_V1,
    "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
}


class OrganDataset(Dataset):
    def __init__(self, path, split):
        with np.load(path) as d:
            self.images = d[f"{split}_images"].copy()
            self.labels = d[f"{split}_labels"].ravel().copy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Transfer small raw images; resize/normalize a whole batch on the GPU.
        return torch.from_numpy(self.images[index]).unsqueeze(0), int(self.labels[index])


def preprocess(images, device):
    images = images.to(device, non_blocking=True).float().div_(255)
    images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
    images = images.expand(-1, 3, -1, -1)
    mean = images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (images - mean) / std


def build_model(kind, pretrained=True):
    model = getattr(models, kind)(weights=WEIGHTS[kind] if pretrained else None)
    if kind == "vit_b_16":
        model.heads.head = nn.Linear(model.heads.head.in_features, 11)
        head = model.heads
    else:
        model.fc = nn.Linear(model.fc.in_features, 11)
        head = model.fc
    return model, head


def step(model, batch, device):
    images = preprocess(batch[0], device)
    labels = batch[1].to(device, non_blocking=True)
    logits = model(images)
    return F.cross_entropy(logits, labels), len(labels), int((logits.argmax(1) == labels).sum())


def run(config, device):
    settings = config["vision"]
    if settings["image_size"] != 224:
        raise ValueError("This ViT pretrained checkpoint requires 224x224 inputs.")
    folder = ROOT / "quiz1" / "Data" / "organamnist"
    manifest = read_json(folder / "manifest.json")
    datasets = {s: OrganDataset(folder / "organamnist.npz", s) for s in ("train", "val", "test")}
    loaders = {s: loader(d, config, settings["batch_size"], shuffle=s == "train") for s, d in datasets.items()}
    comparison = []
    for kind in ("vit_b_16", "resnet50"):
        seed_all(config["seed"])
        model, head = build_model(kind)
        model = model.to(device)
        directory = ROOT / "quiz3" / "Output" / kind
        identity = {"model": kind, "seed": config["seed"], "data": digest(manifest),
                    "weights": str(WEIGHTS[kind]),
                    "preprocessing": "gray-to-RGB/224/tensor-bilinear-antialias/imagenet/no augmentation"}
        for p in model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
        warm = dict(settings, lr=settings["warmup_lr"], epochs=settings["warmup_epochs"],
                    patience=settings["warmup_epochs"] + 1)
        # Keep frozen feature extractor (especially ResNet BatchNorm) in eval mode.
        def warm_step(m, batch, d):
            if m.training:
                m.eval()
                head.train()
            return step(m, batch, d)
        warm_done = fit(model, loaders["train"], loaders["val"], warm_step, warm,
                        directory / "warmup", device, {**identity, "phase": "head"})
        for p in model.parameters():
            p.requires_grad = True
        done = fit(model, loaders["train"], loaders["val"], step, settings, directory, device,
                   {**identity, "phase": "full_finetune"})
        model.eval()
        first = preprocess(next(iter(loaders["test"]))[0], device)
        probs = []
        with torch.inference_mode(), amp_context(device):
            for _ in range(3):
                model(first)
            synchronize(device)
            tick = time.perf_counter()
            for images, _ in loaders["test"]:
                probs.append(model(preprocess(images, device)).float().softmax(1).cpu().numpy())
            synchronize(device)
        seconds = time.perf_counter() - tick
        metrics = classification_metrics(datasets["test"].labels, np.concatenate(probs),
                                         manifest["classes"], directory)
        p = np.concatenate(probs)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        indices = list(np.flatnonzero(p.argmax(1) == datasets["test"].labels)[:6])
        indices += list(np.flatnonzero(p.argmax(1) != datasets["test"].labels)[:6])
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, i in zip(axes.ravel(), indices):
            ax.imshow(datasets["test"].images[i], cmap="gray")
            ax.set_title(f"True: {manifest['classes'][datasets['test'].labels[i]]}\n"
                         f"Pred: {manifest['classes'][p[i].argmax()]}", fontsize=9)
        fig.tight_layout()
        fig.savefig(directory / "prediction_examples.png", dpi=150)
        plt.close(fig)
        write_json(directory / "timing.json", {"end_to_end_test_seconds": seconds,
                   "images_per_second": len(p) / seconds, "batch_size": settings["batch_size"],
                   "includes": "loading + transfer + forward + CPU probabilities"})
        comparison.append({"model": kind, "top1": metrics["top1"], "top5": metrics["top5"],
                           "macro_f1": metrics["macro_f1"], "macro_auc": metrics["macro_auc"],
                           "parameters": done["parameters"],
                           "training_seconds": done["training_seconds"] + warm_done["training_seconds"],
                           "test_seconds": seconds, "best_epoch": done["best_epoch"]})
        del model, head, first
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_csv(ROOT / "quiz3" / "Output" / "comparison.csv", comparison)
    winner = max(comparison, key=lambda r: r["macro_auc"])
    report("quiz3", "Quiz 3：ViT 與 ResNet 微調",
           "## 方法\n\nViT-B/16 與 ResNet-50 使用 ImageNet-1K V1 預訓練權重，"
           "分類層改為 11 類；先訓練 head，再以較小 learning rate 微調全模型。"
           "使用相同官方 train/val/test、RGB 三通道、224×224 與 ImageNet 正規化，不使用隨機擴增。"
           "AdamW、AMP、梯度裁切、驗證 Accuracy 選權重及 early stopping。\n\n"
           "## 測試結果\n\n" + table(comparison) +
           f"\n\n本次 Macro-AUC 較高者為 {winner['model']}。"
           "Macro-AUC 是各器官 one-vs-rest ROC-AUC 的算術平均，輸入為 softmax 機率。"
           "Accuracy 與 AUC 衡量面向不同，應同時閱讀混淆矩陣與各類 ROC。\n\n"
           "## 檔案與限制\n\n各模型目錄保存權重、完整逐 epoch 記錄、ROC、各類 AUC、"
           "混淆矩陣、逐筆機率與 prediction_examples.png。"
           "放大 28×28 CT 不會增加細節；單一 seed 比較不宣稱統計顯著性。"
           "training_seconds 含 head warmup，test_seconds 包含資料載入與傳輸。")
    return comparison


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.rtx5090.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = read_json(args.config)
    run(config, configure(config, args.device))
