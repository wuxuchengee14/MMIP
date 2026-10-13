from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)

from common.utils import write_csv, write_json


def classification_metrics(labels, probs, names, directory, sample_ids=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    y, p = np.asarray(labels), np.asarray(probs)
    if p.shape != (len(y), len(names)) or not np.allclose(p.sum(1), 1, atol=1e-4):
        raise ValueError("Invalid probability array")
    pred = p.argmax(1)
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "top1": float((pred == y).mean()),
        "top5": float((np.argsort(p, axis=1)[:, -min(5, len(names)):] == y[:, None]).any(1).mean()),
        "test_samples": len(y),
    }
    aucs = []
    fig, ax = plt.subplots(figsize=(8, 6))
    roc_rows = []
    for i, name in enumerate(names):
        binary = (y == i).astype(int)
        if len(np.unique(binary)) < 2:
            raise ValueError(f"Test split missing positive or negative examples of {name}")
        fpr, tpr, threshold = roc_curve(binary, p[:, i])
        auc = float(roc_auc_score(binary, p[:, i]))
        aucs.append(auc)
        ax.plot(fpr, tpr, label=f"{name}: {auc:.4f}")
        roc_rows.append({"class": name, "auc": auc})
        write_csv(directory / f"roc_class_{i}.csv", [
            {"fpr": float(a), "tpr": float(b), "threshold": float(c) if np.isfinite(c) else "inf"}
            for a, b, c in zip(fpr, tpr, threshold)
        ])
    metrics["macro_auc"] = float(np.mean(aucs))
    if len(names) == 2:
        metrics["positive_auc"] = aucs[1]
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="One-vs-rest ROC")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "roc.png", dpi=150)
    plt.close(fig)
    cm = confusion_matrix(y, pred, labels=np.arange(len(names)))
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=np.arange(len(names)), yticks=np.arange(len(names)),
           xticklabels=names, yticklabels=names, xlabel="Predicted", ylabel="True")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(directory / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    write_csv(directory / "confusion_matrix.csv",
              [{"true_class": name, **{n: int(cm[i, j]) for j, n in enumerate(names)}} for i, name in enumerate(names)])
    ids = sample_ids if sample_ids is not None else range(len(y))
    write_csv(directory / "predictions.csv", [
        {"id": sid, "true": names[int(a)], "predicted": names[int(b)],
         **{f"prob_{name}": float(prob[k]) for k, name in enumerate(names)}}
        for sid, a, b, prob in zip(ids, y, pred, p)
    ])
    np.savez_compressed(directory / "probabilities.npz", labels=y, probabilities=p)
    write_csv(directory / "class_auc.csv", roc_rows)
    write_json(directory / "classification_report.json",
               classification_report(y, pred, target_names=names, labels=np.arange(len(names)),
                                     output_dict=True, zero_division=0))
    write_json(directory / "metrics.json", metrics)
    return metrics
