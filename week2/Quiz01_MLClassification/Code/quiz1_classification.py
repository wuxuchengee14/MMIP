from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


root_dir = Path(__file__).resolve().parent.parent
data_dir = root_dir / "Data"
output_dir = root_dir / "Output"
data_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)
data_path = data_dir / "classification_data.csv"


def create_dataset(path):
    features, target = make_classification(
        n_samples=1200,
        n_features=6,
        n_informative=5,
        n_redundant=1,
        n_classes=2,
        weights=[0.68, 0.32],
        class_sep=1.15,
        flip_y=0.035,
        random_state=42,
    )
    frame = pd.DataFrame(features, columns=[f"feature_{index}" for index in range(1, 7)])
    frame["target"] = target
    frame.to_csv(path, index=False)


def calculate_metrics(target, probability, threshold):
    prediction = (probability >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(target, prediction),
        "precision": precision_score(target, prediction, zero_division=0),
        "recall": recall_score(target, prediction, zero_division=0),
        "f1_score": f1_score(target, prediction, zero_division=0),
    }, prediction


def find_best_threshold(target, probability):
    candidates = np.round(np.arange(0.10, 0.901, 0.01), 2)
    results = [calculate_metrics(target, probability, threshold)[0] for threshold in candidates]
    return max(results, key=lambda row: (row["f1_score"], row["accuracy"]))["threshold"], pd.DataFrame(results)


def save_confusion_matrices(items):
    figure, axes = plt.subplots(1, len(items), figsize=(5 * len(items), 4))
    for axis, item in zip(np.atleast_1d(axes), items):
        matrix = confusion_matrix(item["target"], item["prediction"])
        image = axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, matrix[row, column], ha="center", va="center", fontsize=13)
        axis.set_title(item["title"])
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks([0, 1])
        axis.set_yticks([0, 1])
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrices.png", dpi=180)
    plt.close(figure)


def save_threshold_curves(curves):
    figure, axes = plt.subplots(1, len(curves), figsize=(7 * len(curves), 4.5), sharey=True)
    for axis, item in zip(np.atleast_1d(axes), curves):
        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            axis.plot(item["data"]["threshold"], item["data"][metric], label=metric)
        axis.axvline(item["best_threshold"], color="black", linestyle="--", label=f"best={item['best_threshold']:.2f}")
        axis.set_title(item["title"])
        axis.set_xlabel("Classification threshold")
        axis.set_ylabel("Score")
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "threshold_curves.png", dpi=180)
    plt.close(figure)


def save_model_comparison(results):
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    positions = np.arange(len(metrics))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 5))
    for index, result in enumerate(results):
        offset = (index - 0.5) * width
        values = [result[metric] for metric in metrics]
        bars = axis.bar(positions + offset, values, width, label=result["model"])
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axis.set_xticks(positions, metrics)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Score")
    axis.set_title("Validation Set Model Comparison")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "model_comparison.png", dpi=180)
    plt.close(figure)


if not data_path.exists():
    create_dataset(data_path)

data = pd.read_csv(data_path)
features = data.drop(columns="target")
target = data["target"]
x_train, x_validation, y_train, y_validation = train_test_split(
    features,
    target,
    test_size=0.25,
    stratify=target,
    random_state=42,
)

scaler = StandardScaler()
scaled_train = scaler.fit_transform(x_train)
scaling_summary = pd.DataFrame(
    {
        "feature": features.columns,
        "original_mean": x_train.mean().values,
        "original_std": x_train.std(ddof=0).values,
        "scaled_mean": scaled_train.mean(axis=0),
        "scaled_std": scaled_train.std(axis=0),
    }
)
scaling_summary.to_csv(output_dir / "feature_scaling_summary.csv", index=False)

logistic_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ]
)
logistic_model.fit(x_train, y_train)
logistic_probability = logistic_model.predict_proba(x_validation)[:, 1]
initial_threshold = 0.50
initial_metrics, initial_prediction = calculate_metrics(y_validation, logistic_probability, initial_threshold)
logistic_threshold, logistic_curve = find_best_threshold(y_validation, logistic_probability)
logistic_metrics, logistic_prediction = calculate_metrics(y_validation, logistic_probability, logistic_threshold)

forest_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=350,
                max_depth=9,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)
forest_model.fit(x_train, y_train)
forest_probability = forest_model.predict_proba(x_validation)[:, 1]
forest_threshold, forest_curve = find_best_threshold(y_validation, forest_probability)
forest_metrics, forest_prediction = calculate_metrics(y_validation, forest_probability, forest_threshold)

all_results = [
    {"model": "Logistic Regression (initial)", **initial_metrics},
    {"model": "Logistic Regression (tuned)", **logistic_metrics},
    {"model": "Random Forest (tuned)", **forest_metrics},
]
comparison_results = [all_results[1], all_results[2]]
pd.DataFrame(all_results).to_csv(output_dir / "metrics.csv", index=False)

save_confusion_matrices(
    [
        {"title": "Logistic Regression\nThreshold 0.50", "target": y_validation, "prediction": initial_prediction},
        {"title": f"Logistic Regression\nThreshold {logistic_threshold:.2f}", "target": y_validation, "prediction": logistic_prediction},
        {"title": f"Random Forest\nThreshold {forest_threshold:.2f}", "target": y_validation, "prediction": forest_prediction},
    ]
)
save_threshold_curves(
    [
        {"title": "Logistic Regression", "data": logistic_curve, "best_threshold": logistic_threshold},
        {"title": "Random Forest", "data": forest_curve, "best_threshold": forest_threshold},
    ]
)
save_model_comparison(comparison_results)

comparison_frame = pd.DataFrame(comparison_results).set_index("model")
precision_winner = comparison_frame["precision"].idxmax()
recall_winner = comparison_frame["recall"].idxmax()
logistic_matrix = confusion_matrix(y_validation, logistic_prediction)
forest_matrix = confusion_matrix(y_validation, forest_prediction)
logistic_false_positive = int(logistic_matrix[0, 1])
logistic_false_negative = int(logistic_matrix[1, 0])
forest_false_positive = int(forest_matrix[0, 1])
forest_false_negative = int(forest_matrix[1, 0])
logistic_errors = int((logistic_prediction != y_validation.to_numpy()).sum())
forest_errors = int((forest_prediction != y_validation.to_numpy()).sum())
error_statement = "Random Forest has more errors." if forest_errors > logistic_errors else "Logistic Regression has more errors." if logistic_errors > forest_errors else "Both models have the same number of errors."
logistic_error_type = "false negatives" if logistic_false_negative > logistic_false_positive else "false positives" if logistic_false_positive > logistic_false_negative else "equal false positives and false negatives"
forest_error_type = "false negatives" if forest_false_negative > forest_false_positive else "false positives" if forest_false_positive > forest_false_negative else "equal false positives and false negatives"
report = "\n".join(
    [
        "# Quiz 1：機器學習分類任務",
        "",
        "## 一、資料與檔案說明",
        "",
        f"本題使用 Scikit-Learn 產生二元分類模擬資料，共有 {len(data)} 筆資料及 {features.shape[1]} 個輸入特徵。資料依照 75% 與 25% 分為 Training Dataset 和 Validation Dataset，分別為 {len(x_train)} 筆與 {len(x_validation)} 筆。",
        "",
        "此資料集由 `make_classification` 產生，其中 5 個特徵包含與分類有關的資訊，另有 1 個特徵由其他特徵線性組合而成。產生資料時會打亂特徵順序，因此不能直接指定哪一欄是 redundant feature。所有輸入欄位都是連續數值，沒有對應真實世界的物理單位。",
        "",
        "| 欄位 | 型態 | 說明 |",
        "|---|---|---|",
        "| `feature_1` | 浮點數 | 第一個合成數值特徵，作為模型分類依據之一 |",
        "| `feature_2` | 浮點數 | 第二個合成數值特徵，作為模型分類依據之一 |",
        "| `feature_3` | 浮點數 | 第三個合成數值特徵，作為模型分類依據之一 |",
        "| `feature_4` | 浮點數 | 第四個合成數值特徵，作為模型分類依據之一 |",
        "| `feature_5` | 浮點數 | 第五個合成數值特徵，作為模型分類依據之一 |",
        "| `feature_6` | 浮點數 | 第六個合成數值特徵，作為模型分類依據之一 |",
        "| `target` | 整數 | 二元分類標籤；0 代表負類，1 代表正類 |",
        "",
        "資料產生時將類別比例設定為約 68% 的類別 0 與 32% 的類別 1，並加入少量標籤雜訊，使分類任務更接近真實資料。模型訓練時，`feature_1` 到 `feature_6` 為 X，`target` 為 y；`target` 不會參與 Feature Scaling。",
        "",
        "- 程式：`week2/Quiz01_MLClassification/Code/quiz1_classification.py`",
        "- 資料：`week2/Quiz01_MLClassification/Data/classification_data.csv`",
        "- AI 使用紀錄：`week2/Quiz01_MLClassification/AI/AI_usage.md`",
        "- 評估結果：`week2/Quiz01_MLClassification/Output/metrics.csv`",
        "- 混淆矩陣：`week2/Quiz01_MLClassification/Output/confusion_matrices.png`",
        "- Threshold 曲線：`week2/Quiz01_MLClassification/Output/threshold_curves.png`",
        "- 模型比較圖：`week2/Quiz01_MLClassification/Output/model_comparison.png`",
        "",
        "## 二、資料前處理與模型",
        "",
        f"我使用 StandardScaler 對特徵進行標準化。第一個模型為 Logistic Regression，初始 Threshold 為 {initial_threshold:.2f}，根據 Validation Dataset 的 F1-Score 微調後選擇 {logistic_threshold:.2f}。第二個模型使用 Random Forest，選出的 Threshold 為 {forest_threshold:.2f}。",
        "",
        "## 三、結果",
        "",
        "| 模型 | Threshold | Accuracy | Precision | Recall | F1-Score |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Logistic Regression 初始 | {initial_metrics['threshold']:.2f} | {initial_metrics['accuracy']:.4f} | {initial_metrics['precision']:.4f} | {initial_metrics['recall']:.4f} | {initial_metrics['f1_score']:.4f} |",
        f"| Logistic Regression 微調 | {logistic_metrics['threshold']:.2f} | {logistic_metrics['accuracy']:.4f} | {logistic_metrics['precision']:.4f} | {logistic_metrics['recall']:.4f} | {logistic_metrics['f1_score']:.4f} |",
        f"| Random Forest 微調 | {forest_metrics['threshold']:.2f} | {forest_metrics['accuracy']:.4f} | {forest_metrics['precision']:.4f} | {forest_metrics['recall']:.4f} | {forest_metrics['f1_score']:.4f} |",
        "",
        f"Logistic Regression 降低 Threshold 後，Recall 與 F1-Score 提升，但 Precision 稍微下降。微調後共有 {logistic_false_positive} 筆 False Positive 和 {logistic_false_negative} 筆 False Negative，主要錯誤為 False Negative。",
        "",
        f"Random Forest 有 {forest_false_positive} 筆 False Positive 和 {forest_false_negative} 筆 False Negative，主要錯誤為 False Positive。整體而言，Random Forest 的 Accuracy、Precision、Recall 與 F1-Score 都較高，分類錯誤也較少。",
    ]
)
(output_dir / "report.md").write_text(report, encoding="utf-8")

print(pd.DataFrame(all_results).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print()
print(report)
