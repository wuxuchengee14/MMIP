"""Download the official 28x28 dataset and persist development-only folds."""
from pathlib import Path
import argparse
import hashlib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from medmnist import INFO, OrganAMNIST
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from common.data import load_arrays
from common.utils import CLASSES, ROOT, file_hash, load_config, write_json


def build_folds(labels, n_splits, seed, groups=None):
    labels = np.asarray(labels)
    if np.bincount(labels, minlength=11).min() < n_splits:
        raise ValueError("Each class must contain at least n_splits samples.")
    if groups is not None and len(np.unique(groups)) < len(groups):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(np.zeros(len(labels)), labels, groups))
        method = "StratifiedGroupKFold (exact-image groups, not patient groups)"
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(np.zeros(len(labels)), labels))
        method = "StratifiedKFold (image level)"
    seen = []
    for train, val in folds:
        if len(np.intersect1d(train, val)) or len(train) + len(val) != len(labels):
            raise ValueError("Invalid fold partition.")
        if groups is not None and set(groups[train]) & set(groups[val]):
            raise ValueError("Exact-image groups overlap across a fold.")
        if len(np.unique(labels[val])) != 11 or len(np.unique(labels[train])) != 11:
            raise ValueError("A fold is missing classes; choose fewer folds.")
        seen.extend(val.tolist())
    if sorted(seen) != list(range(len(labels))):
        raise ValueError("Every development sample must be validated exactly once.")
    return folds, method


def image_hashes(images):
    return np.array([hashlib.sha256(image.tobytes()).hexdigest() for image in images])


def prepare(config, root=ROOT, download=True, check_md5=True):
    root = Path(root)
    data_dir = root / "Quiz01_DatasetPreparation" / "Data"
    output = root / "Quiz01_DatasetPreparation" / "Output"
    data_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if download:
        OrganAMNIST(split="train", root=str(data_dir), size=28, download=True)
    path = data_dir / "organamnist.npz"
    info = INFO["organamnist"]
    if check_md5 and file_hash(path, "md5") != info["MD5"]:
        raise ValueError("OrganAMNIST does not match the official MD5. Download the official file again.")
    images, labels, test_images, test_labels = load_arrays(root)
    if images.shape[1:] != (28, 28) or test_images.shape[1:] != (28, 28):
        raise ValueError("This experiment requires the original 28x28 grayscale dataset.")
    for values in (labels, test_labels):
        if values.min() < 0 or values.max() > 10:
            raise ValueError("Expected class labels 0..10.")
    if check_md5 and (len(labels) != 41052 or len(test_labels) != 17778):
        raise ValueError("Unexpected official dataset sizes.")
    # Hashes group identical development images without changing the official test set.
    groups, test_groups = image_hashes(images), image_hashes(test_images)
    folds, method = build_folds(labels, config["folds"], config["seed"], groups)
    arrays = {}
    rows = []
    for fold, (train, val) in enumerate(folds):
        arrays[f"train_{fold}"], arrays[f"val_{fold}"] = train, val
        for split, ids in (("train", train), ("val", val)):
            counts = np.bincount(labels[ids], minlength=11)
            rows.extend({"fold": fold + 1, "split": split, "class": name, "count": int(counts[k])}
                        for k, name in enumerate(CLASSES))
    split_path = data_dir / "folds.npz"
    np.savez_compressed(split_path, **arrays)
    pd.DataFrame(rows).to_csv(output / "fold_class_counts.csv", index=False)
    with np.load(path, allow_pickle=False) as raw:
        counts = {split: np.bincount(raw[f"{split}_labels"].reshape(-1), minlength=11)
                  for split in ("train", "val", "test")}
        provenance = ([f"train_{i}" for i in range(len(raw["train_labels"]))] +
                      [f"val_{i}" for i in range(len(raw["val_labels"]))])
    pd.DataFrame({"development_index": np.arange(len(labels)), "source_id": provenance,
                  "label": labels, "class": [CLASSES[k] for k in labels]}).to_csv(
                      output / "development_manifest.csv", index=False)
    pd.DataFrame(counts, index=CLASSES).to_csv(output / "official_class_counts.csv")
    fig, ax = plt.subplots(figsize=(12, 5))
    pd.DataFrame(counts, index=CLASSES).plot.bar(ax=ax)
    ax.set(title="OrganAMNIST official class distribution", ylabel="Images")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output / "class_distribution.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for k, name in enumerate(CLASSES):
        ax = axes.flat[k]
        ax.imshow(images[np.flatnonzero(labels == k)[0]], cmap="gray", vmin=0, vmax=255)
        ax.set_title(name)
        ax.axis("off")
    axes.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(output / "class_examples.png", dpi=160)
    plt.close(fig)
    overlap = set(groups) & set(test_groups)
    manifest = {"dataset": "OrganAMNIST", "source": info["url"], "description": info["description"],
                "license": info["license"], "classes": CLASSES, "original_image_size": 28,
                "official_counts": {k: int(v.sum()) for k, v in counts.items()},
                "development_samples": len(labels), "test_samples": len(test_labels),
                "seed": config["seed"], "folds": config["folds"], "method": method,
                "patient_grouping_available": False,
                "limitation": "The distributed NPZ has no patient/scan IDs. CV is not patient-independent.",
                "duplicate_audit": {"development_duplicate_images": len(groups) - len(set(groups)),
                                    "unique_hashes_shared_with_official_test": len(overlap),
                                    "action": "Keep official test unchanged; group exact development duplicates in CV."},
                "dataset_sha256": file_hash(path), "folds_sha256": file_hash(split_path)}
    write_json(data_dir / "split_manifest.json", manifest)
    report = f"""# Quiz 1：OrganAMNIST 資料準備

輸入為 28×28 腹部 CT 灰階軸向器官影像，輸出為 11 個器官類別之一。
來源：MedMNIST / LiTS；授權：{info['license']}。

官方 train={counts['train'].sum()}、validation={counts['val'].sum()}、test={counts['test'].sum()}。
合併 train 與 validation 共 {len(labels)} 張，採用 {config['folds']}-fold，seed={config['seed']}。
切分方法：{method}。Test 保留官方切分，未參與 fold、超參數選擇或 early stopping。

標準 NPZ 無病例 ID，因此無法保證 CV 病例獨立，應以官方 test 評估泛化。
開發集重複影像數：{manifest['duplicate_audit']['development_duplicate_images']}；
開發集與官方 test 共用的不同影像雜湊數：{len(overlap)}。
若存在官方跨集重複，保留官方測試集並揭露此限制。

類別數量、影像範例、fold 分布與原始索引對應見同資料夾 CSV / PNG。
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    print(f"Prepared {config['folds']} folds; development={len(labels)}, test={len(test_labels)}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--no-download", action="store_true", help="Use Quiz01_DatasetPreparation/Data/organamnist.npz already present.")
    args = parser.parse_args()
    prepare(load_config(args.config), download=not args.no_download)


if __name__ == "__main__":
    main()
