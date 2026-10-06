"""Visualize two trained kernels and Grad-CAM for both trained models."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from common.data import data_directory, load_arrays, make_transform
from common.train import load_trained
from common.utils import CLASSES, ROOT, file_hash, get_device, read_json, seed_everything, write_json


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.activation = self.gradient = None
        self.handle = layer.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self.activation = output.detach()
        output.register_hook(self._gradient)

    def _gradient(self, gradient):
        self.gradient = gradient.detach()

    def __call__(self, image, target=None):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        image = image.detach().requires_grad_(True)
        logits = self.model(image)
        if len(logits) != 1:
            raise ValueError("GradCAM expects one image.")
        target = int(logits.argmax(dim=1).item()) if target is None else int(target)
        logits[0, target].backward()
        weights = self.gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activation).sum(dim=1, keepdim=True).relu()
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam -= cam.min()
        maximum = float(cam.max())
        if maximum > 0:
            cam /= maximum
        return cam.detach().cpu().numpy(), logits.detach().softmax(dim=1)[0].cpu().numpy()

    def close(self):
        self.handle.remove()


def gradcam_cases(labels, probabilities):
    predicted = probabilities.argmax(axis=1)
    cases = []
    for category in range(11):
        for correct in (True, False):
            choices = np.flatnonzero((labels == category) & ((predicted == labels) == correct))
            if len(choices):
                # Deterministic, first sample in each true-class/outcome category.
                cases.append(int(choices[0]))
    return cases


def check_artifact(meta, root):
    data_dir = data_directory(root)
    manifest = read_json(data_dir / "split_manifest.json")
    from common.experiments import identity
    if file_hash(data_dir / "organamnist.npz") != manifest["dataset_sha256"]:
        raise ValueError("Dataset has changed since training.")
    if meta["dataset_signature"] != identity(meta["config"], manifest):
        raise ValueError("Checkpoint does not match dataset/splits.")


def visualize_kernels(model, meta, images, labels, output, device, kernel_ids=(0, 1)):
    conv = next(module for module in model.modules() if isinstance(module, nn.Conv2d))
    if len(kernel_ids) != 2 or len(set(kernel_ids)) != 2 or any(i < 0 or i >= conv.out_channels for i in kernel_ids):
        raise ValueError("Choose exactly two different first-layer output channel indices.")
    sample = int(np.flatnonzero(labels == 6)[0])  # Fixed liver example from development pool.
    image = make_transform("plain", meta["config"])(Image.fromarray(images[sample])).unsqueeze(0).to(device)
    with torch.no_grad():
        maps = conv(image)[0].cpu().numpy()
    weights = conv.weight.detach().cpu().numpy()
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    rows = []
    for row, index in enumerate(kernel_ids):
        kernel = weights[index, 0]
        bound = max(float(np.abs(kernel).max()), 1e-8)
        axes[row, 0].imshow(images[sample], cmap="gray")
        axes[row, 0].set_title("Original liver example")
        chart = axes[row, 1].imshow(kernel, cmap="coolwarm", vmin=-bound, vmax=bound)
        axes[row, 1].set_title(f"Trained kernel {index} (3x3 weights)")
        fig.colorbar(chart, ax=axes[row, 1], fraction=0.046)
        chart = axes[row, 2].imshow(maps[index], cmap="coolwarm")
        axes[row, 2].set_title("Response before ReLU")
        fig.colorbar(chart, ax=axes[row, 2], fraction=0.046)
        for ax in axes[row]:
            ax.axis("off")
        rows.append({"kernel": index, "weight_mean": float(kernel.mean()),
                     "weight_std": float(kernel.std()),
                     "horizontal_weight_difference": float(np.abs(np.diff(kernel, axis=1)).mean()),
                     "vertical_weight_difference": float(np.abs(np.diff(kernel, axis=0)).mean())})
    fig.tight_layout()
    fig.savefig(output / "kernels_and_responses.png", dpi=180)
    plt.close(fig)
    write_json(output / "kernel_statistics.json", rows)
    np.savez_compressed(output / "kernel_weights.npz", **{f"kernel_{i}": weights[i] for i in kernel_ids})
    lines = ["# 兩個訓練後 Kernel 的分析", "",
             "選擇 Plain CNN 第一層的兩個固定 kernel，熱圖顯示權重，右側是同一張肝臟影像的卷積反應（ReLU 前）。",
             "正負權重相鄰的結構可能對局部灰階差異或邊緣有反應；方向需結合反應圖判斷，不能視為已證實的器官偵測器。", ""]
    for row in rows:
        lines.append(f"- Kernel {row['kernel']}：權重平均 {row['weight_mean']:.5f}、標準差 {row['weight_std']:.5f}；"
                     f"水平／垂直相鄰權重平均差 {row['horizontal_weight_difference']:.5f} / {row['vertical_weight_difference']:.5f}。")
    lines.extend(["", "請結合熱圖中正負區塊的排列，以及特徵圖中實際強烈回應的位置，描述兩個 kernel 的差異。"])
    (output / "kernel_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def explain_models(device, root=ROOT, quiz=3, kernel_ids=(0, 1), checkpoint_subdir="final"):
    root = Path(root)
    images, labels, test_images, _ = load_arrays(root)
    quiz_directories = {2: "Quiz02_CNNClassification", 3: "Quiz03_Generalization"}
    output = root / "Quiz03_Generalization" / "Output" / "explanations" / f"quiz{quiz}"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("plain", "vgg19"):
        base = root / quiz_directories[quiz] / "Output" / name
        model, meta, _ = load_trained(base / checkpoint_subdir / "best.pt", device)
        check_artifact(meta, root)
        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        if name == "plain":
            visualize_kernels(model, meta, images, labels, folder, device, kernel_ids)
        with np.load(base / "test" / "predictions.npz", allow_pickle=False) as saved:
            y, p, ids = saved["labels"], saved["probabilities"], saved["indices"]
        last_conv = [layer for layer in model.modules() if isinstance(layer, nn.Conv2d)][-1]
        # Only input gradients are needed; avoid allocating VGG classifier parameter gradients.
        for parameter in model.parameters():
            parameter.requires_grad = False
        cam_engine = GradCAM(model, last_conv)
        transform = make_transform(name, meta["config"])
        records = []
        try:
            for position in gradcam_cases(y, p):
                index = int(ids[position])
                image = transform(Image.fromarray(test_images[index])).unsqueeze(0).to(device)
                target = int(p[position].argmax())
                cam, probabilities = cam_engine(image, target)
                border = max(1, cam.shape[0] // 10)
                mask = np.ones(cam.shape, dtype=bool)
                mask[border:-border, border:-border] = False
                mass = float(cam.sum())
                border_fraction = float(cam[mask].sum() / mass) if mass > 0 else None
                original = np.asarray(Image.fromarray(test_images[index]).resize(
                    (meta["config"]["input_size"],) * 2, Image.Resampling.BILINEAR))
                fig, axes = plt.subplots(1, 3, figsize=(11, 4))
                axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
                axes[0].set_title("Input (upsampled from 28x28)")
                axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
                axes[1].set_title("Grad-CAM: predicted class")
                axes[2].imshow(original, cmap="gray", vmin=0, vmax=255)
                axes[2].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
                axes[2].set_title("Overlay")
                for ax in axes:
                    ax.axis("off")
                correct = target == int(y[position])
                fig.suptitle(f"{name} | true={CLASSES[y[position]]} | predicted={CLASSES[target]} | "
                             f"p={probabilities[target]:.3f} | correct={correct}")
                fig.tight_layout()
                fig.savefig(folder / f"gradcam_test_{index}.png", dpi=160)
                plt.close(fig)
                np.save(folder / f"gradcam_test_{index}.npy", cam)
                records.append({"test_index": index, "true_class": CLASSES[y[position]],
                                "predicted_class": CLASSES[target], "correct": correct,
                                "confidence": float(probabilities[target]), "zero_cam": mass == 0,
                                "outer_10_percent_cam_mass": border_fraction})
        finally:
            cam_engine.close()
        pd.DataFrame(records).to_csv(folder / "gradcam_cases.csv", index=False)
        text = [f"# {name} Grad-CAM 分析", "",
                "每類選取第一個正確與第一個錯誤案例；某類沒有錯誤案例時不製造案例。",
                "熱圖解釋的是預測類別，最後卷積層的粗略關注區域被上採樣至輸入大小。",
                "原始 28×28 影像沒有器官分割標註，無法量化關注區域是否精確對應器官。", ""]
        for record in records:
            fraction = record["outer_10_percent_cam_mass"]
            fraction_text = "無有效正向 CAM" if fraction is None else f"外圍 10% 邊帶占熱圖總量 {fraction:.1%}"
            text.append(f"- test_{record['test_index']}：{record['true_class']} → {record['predicted_class']}，"
                        f"信心 {record['confidence']:.3f}，{fraction_text}。"
                        + ("觀察錯誤案例是否偏向背景或相似器官紋理。" if not record["correct"] else "對照輪廓與內部紋理檢視關注位置。"))
        text.extend(["", "邊帶比例僅是背景偏向的診斷線索；並非器官定位分數或因果證據。"])
        (folder / "xai_analysis.md").write_text("\n".join(text), encoding="utf-8")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quiz", type=int, choices=[2, 3], default=3, help="Choose baseline or augmented checkpoints.")
    parser.add_argument("--kernels", nargs=2, type=int, default=[0, 1])
    args = parser.parse_args()
    seed_everything(42)
    explain_models(get_device(args.device), quiz=args.quiz, kernel_ids=args.kernels)


if __name__ == "__main__":
    main()
