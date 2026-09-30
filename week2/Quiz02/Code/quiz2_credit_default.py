from copy import deepcopy
from pathlib import Path
import warnings
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


root_dir = Path(__file__).resolve().parent.parent
data_dir = root_dir / "Data"
output_dir = root_dir / "Output"
data_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)
data_path = data_dir / "UCI_Credit_Card.csv"
archive_path = data_dir / "credit-card-default.zip"
target_name = "default.payment.next.month"
epochs = 60
batch_size = 4096
seed = 42


if not data_path.exists() and archive_path.exists():
    with ZipFile(archive_path) as archive:
        archive.extract("UCI_Credit_Card.csv", data_dir)

if not data_path.exists():
    raise FileNotFoundError("Place UCI_Credit_Card.csv in the Data directory.")

np.random.seed(seed)
torch.manual_seed(seed)
warnings.filterwarnings("ignore", message="(?s).*is not compatible with the current PyTorch installation.*")
cuda_capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
cuda_architecture = f"sm_{cuda_capability[0]}{cuda_capability[1]}" if cuda_capability else None
cuda_supported = torch.cuda.is_available() and cuda_architecture in torch.cuda.get_arch_list()
if cuda_supported:
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if cuda_supported else "cpu")
data = pd.read_csv(data_path)
features = data.drop(columns=["ID", target_name])
target = data[target_name].astype(np.float32)
x_train, x_validation, y_train, y_validation = train_test_split(
    features,
    target,
    test_size=0.20,
    stratify=target,
    random_state=seed,
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
x_validation_scaled = scaler.transform(x_validation).astype(np.float32)
scaling_summary = pd.DataFrame(
    {
        "feature": features.columns,
        "original_mean": x_train.mean().values,
        "original_std": x_train.std(ddof=0).values,
        "scaled_mean": x_train_scaled.mean(axis=0),
        "scaled_std": x_train_scaled.std(axis=0),
    }
)
scaling_summary.to_csv(output_dir / "feature_scaling_summary.csv", index=False)

x_train_tensor = torch.from_numpy(x_train_scaled)
y_train_tensor = torch.from_numpy(y_train.to_numpy()).reshape(-1, 1)
x_validation_tensor = torch.from_numpy(x_validation_scaled)
y_validation_tensor = torch.from_numpy(y_validation.to_numpy()).reshape(-1, 1)
train_generator = torch.Generator().manual_seed(seed)
train_loader = DataLoader(
    TensorDataset(x_train_tensor, y_train_tensor),
    batch_size=batch_size,
    shuffle=True,
    generator=train_generator,
)
validation_loader = DataLoader(
    TensorDataset(x_validation_tensor, y_validation_tensor),
    batch_size=batch_size,
    shuffle=False,
)


class BaselineMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, values):
        return self.network(values)


class ImprovedMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, values):
        return self.network(values)


def epoch_loss(model, loader, loss_function, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    for batch_features, batch_target in loader:
        batch_features = batch_features.to(device)
        batch_target = batch_target.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(batch_features)
            loss = loss_function(logits, batch_target)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        total_loss += loss.item() * batch_features.size(0)
        total_samples += batch_features.size(0)
    return total_loss / total_samples


def train_model(name, model, optimizer, scheduler=None, minimum_epochs=epochs, patience=None):
    positive_count = float(y_train.sum())
    negative_count = float(len(y_train) - positive_count)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative_count / positive_count], device=device))
    history = {"train_loss": [], "validation_loss": [], "learning_rate": []}
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    waiting = 0
    for epoch in range(1, epochs + 1):
        train_loss = epoch_loss(model, train_loader, loss_function, optimizer)
        validation_loss = epoch_loss(model, validation_loader, loss_function)
        current_rate = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["learning_rate"].append(current_rate)
        if scheduler is not None:
            scheduler.step(validation_loss)
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            waiting = 0
        else:
            waiting += 1
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"{name} | epoch {epoch:02d}/{epochs} | train_loss={train_loss:.5f} | validation_loss={validation_loss:.5f} | lr={current_rate:.6f}")
        if patience is not None and epoch >= minimum_epochs and waiting >= patience:
            break
    model.load_state_dict(best_state)
    return history, best_loss


def predict(model, values):
    model.eval()
    with torch.no_grad():
        logits = model(values.to(device))
        return torch.sigmoid(logits).cpu().numpy().reshape(-1)


def evaluate(model, name):
    probability = predict(model, x_validation_tensor)
    prediction = (probability >= 0.50).astype(int)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_validation, prediction),
        "precision": precision_score(y_validation, prediction, zero_division=0),
        "recall": recall_score(y_validation, prediction, zero_division=0),
        "f1_score": f1_score(y_validation, prediction, zero_division=0),
    }
    return metrics, probability, prediction


def save_loss_curves(histories):
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for axis, item in zip(axes, histories):
        epoch_values = np.arange(1, len(item["history"]["train_loss"]) + 1)
        axis.plot(epoch_values, item["history"]["train_loss"], label="Training Loss")
        axis.plot(epoch_values, item["history"]["validation_loss"], label="Validation Loss")
        axis.set_title(item["name"])
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Weighted BCE Loss")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curves.png", dpi=180)
    plt.close(figure)


def save_metric_comparison(results):
    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    positions = np.arange(len(metric_names))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 5))
    for index, result in enumerate(results):
        values = [result[metric] for metric in metric_names]
        bars = axis.bar(positions + (index - 0.5) * width, values, width, label=result["model"])
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axis.set_xticks(positions, metric_names)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Score")
    axis.set_title("Validation Metrics Before and After Improvement")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "metric_comparison.png", dpi=180)
    plt.close(figure)


def save_confusion_matrices(items):
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, item in zip(axes, items):
        matrix = confusion_matrix(y_validation, item["prediction"])
        axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, matrix[row, column], ha="center", va="center", fontsize=13)
        axis.set_title(item["name"])
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks([0, 1])
        axis.set_yticks([0, 1])
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrices.png", dpi=180)
    plt.close(figure)


input_size = x_train_scaled.shape[1]
baseline_model = BaselineMLP(input_size).to(device)
baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001)
baseline_history, baseline_best_loss = train_model("Baseline MLP", baseline_model, baseline_optimizer)

torch.manual_seed(seed)
improved_model = ImprovedMLP(input_size).to(device)
improved_optimizer = torch.optim.AdamW(improved_model.parameters(), lr=0.001, weight_decay=0.0005)
improved_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    improved_optimizer,
    mode="min",
    factor=0.5,
    patience=5,
    min_lr=0.00001,
)
improved_history, improved_best_loss = train_model(
    "Improved MLP",
    improved_model,
    improved_optimizer,
    scheduler=improved_scheduler,
    minimum_epochs=50,
    patience=10,
)

baseline_metrics, baseline_probability, baseline_prediction = evaluate(baseline_model, "Baseline MLP")
improved_metrics, improved_probability, improved_prediction = evaluate(improved_model, "Improved MLP")
results = [baseline_metrics, improved_metrics]
pd.DataFrame(results).to_csv(output_dir / "metrics.csv", index=False)
pd.DataFrame(baseline_history).to_csv(output_dir / "baseline_history.csv", index_label="epoch")
pd.DataFrame(improved_history).to_csv(output_dir / "improved_history.csv", index_label="epoch")
torch.save(baseline_model.state_dict(), output_dir / "baseline_mlp.pt")
torch.save(improved_model.state_dict(), output_dir / "improved_mlp.pt")

sample_position = 0
sample_features = x_validation.iloc[sample_position]
sample_actual = int(y_validation.iloc[sample_position])
sample_probability = float(improved_probability[sample_position])
sample_prediction = int(improved_prediction[sample_position])
sample_result = pd.DataFrame(
    [
        {
            "validation_row_id": int(x_validation.index[sample_position]),
            "actual_class": sample_actual,
            "predicted_class": sample_prediction,
            "default_probability": sample_probability,
            **sample_features.to_dict(),
        }
    ]
)
sample_result.to_csv(output_dir / "sample_prediction.csv", index=False)

save_loss_curves(
    [
        {"name": "Baseline MLP", "history": baseline_history},
        {"name": "Improved MLP", "history": improved_history},
    ]
)
save_metric_comparison(results)
save_confusion_matrices(
    [
        {"name": "Baseline MLP", "prediction": baseline_prediction},
        {"name": "Improved MLP", "prediction": improved_prediction},
    ]
)

baseline_gap = baseline_history["validation_loss"][-1] - baseline_history["train_loss"][-1]
improved_gap = improved_history["validation_loss"][-1] - improved_history["train_loss"][-1]
baseline_stability = float(np.std(baseline_history["validation_loss"][-10:]))
improved_stability = float(np.std(improved_history["validation_loss"][-10:]))
overfitting_observation = "改善版縮小了 Training Loss 與 Validation Loss 的差距，顯示抑制 overfitting 的效果較好。" if improved_gap < baseline_gap else "改善版沒有縮小 Training Loss 與 Validation Loss 的差距。"
loss_observation = "改善版取得較低的最佳 Validation Loss。" if improved_best_loss < baseline_best_loss else "基準模型取得較低的最佳 Validation Loss。"
stability_observation = "改善版最後 10 個 Epoch 的 Validation Loss 較穩定。" if improved_stability < baseline_stability else "基準模型最後 10 個 Epoch 的 Validation Loss 較穩定。"
f1_observation = "改善版提高了 Validation F1-Score。" if improved_metrics["f1_score"] > baseline_metrics["f1_score"] else "改善版沒有提高 Validation F1-Score。"
report = "\n".join(
    [
        "# Quiz 2：深度學習信用卡違約預測",
        "",
        "## 一、資料與檔案說明",
        "",
        f"本題使用 Kaggle 的 Default of Credit Card Clients Dataset，共有 {len(data)} 筆資料及 {input_size} 個輸入特徵。資料依照 80% 與 20% 分為 Training Dataset 和 Validation Dataset，分別為 {len(x_train)} 筆與 {len(x_validation)} 筆。違約類別的比例為 {target.mean():.4f}。",
        "",
        "- 程式：`week2/Quiz02/Code/quiz2_credit_default.py`",
        "- 原始資料：`week2/Quiz02/Data/UCI_Credit_Card.csv`",
        "- Kaggle 壓縮檔：`week2/Quiz02/Data/credit-card-default.zip`",
        "- 套件清單：`week2/Quiz02/requirements.txt`",
        "- AI 使用紀錄：`week2/Quiz02/AI/AI_usage.md`",
        "- Loss 曲線：`week2/Quiz02/Output/loss_curves.png`",
        "- 指標比較圖：`week2/Quiz02/Output/metric_comparison.png`",
        "- 混淆矩陣：`week2/Quiz02/Output/confusion_matrices.png`",
        "- 單筆預測：`week2/Quiz02/Output/sample_prediction.csv`",
        "- 模型權重：`week2/Quiz02/Output/baseline_mlp.pt`、`week2/Quiz02/Output/improved_mlp.pt`",
        "",
        "## 二、資料處理與基準模型",
        "",
        f"我先移除 ID 欄位，再使用 StandardScaler 標準化輸入特徵。基準 MLP 使用 60 Epochs、Batch Size 4096、Learning Rate 0.001、Adam Optimizer，以及 64、32 個神經元的隱藏層。由於資料類別不平衡，Loss Function 使用加權 Binary Cross Entropy。實際運算裝置為 {device}。",
        "",
        "## 三、改善策略",
        "",
        "改善版 MLP 的隱藏層為 128、64、32 個神經元，加入 Batch Normalization、Dropout、L2 Regularization 和 ReduceLROnPlateau，Optimizer 改為 AdamW。Dropout Rate 為 0.30 與 0.20，Weight Decay 為 0.0005。",
        "",
        "## 四、結果與觀察",
        "",
        "| 模型 | Accuracy | Precision | Recall | F1-Score |",
        "|---|---:|---:|---:|---:|",
        f"| Baseline MLP | {baseline_metrics['accuracy']:.4f} | {baseline_metrics['precision']:.4f} | {baseline_metrics['recall']:.4f} | {baseline_metrics['f1_score']:.4f} |",
        f"| Improved MLP | {improved_metrics['accuracy']:.4f} | {improved_metrics['precision']:.4f} | {improved_metrics['recall']:.4f} | {improved_metrics['f1_score']:.4f} |",
        "",
        f"基準模型的最佳 Validation Loss 為 {baseline_best_loss:.6f}，改善版為 {improved_best_loss:.6f}。基準模型最後的 Loss Gap 為 {baseline_gap:.6f}，改善版為 {improved_gap:.6f}。",
        "",
        overfitting_observation,
        loss_observation,
        stability_observation,
        f1_observation,
        "",
        f"程式展示的單筆 Validation Sample 為資料列 {int(x_validation.index[sample_position])}，實際類別為 {sample_actual}，預測類別為 {sample_prediction}，預測違約機率為 {sample_probability:.6f}。完整資料保存在 `week2/Quiz02/Output/sample_prediction.csv`。",
    ]
)
(output_dir / "report.md").write_text(report, encoding="utf-8")

print()
print(pd.DataFrame(results).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print()
print(f"Validation sample prediction | row={int(x_validation.index[sample_position])} | actual={sample_actual} | predicted={sample_prediction} | probability={sample_probability:.6f}")
print()
print(report)
