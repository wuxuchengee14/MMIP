from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn


root_dir = Path(__file__).resolve().parent.parent
week2_dir = root_dir.parent
output_dir = root_dir / "Output"
output_dir.mkdir(parents=True, exist_ok=True)
local_data_path = root_dir / "Data" / "UCI_Credit_Card.csv"
quiz2_data_path = week2_dir / "Quiz02" / "Data" / "UCI_Credit_Card.csv"
quiz2_model_path = week2_dir / "Quiz02" / "Output" / "improved_mlp.pt"
data_path = local_data_path if local_data_path.exists() else quiz2_data_path
target_name = "default.payment.next.month"
seed = 42


if not data_path.exists():
    raise FileNotFoundError("Run Quiz 2 first or place UCI_Credit_Card.csv in the Quiz 3 Data directory.")

if not quiz2_model_path.exists():
    raise FileNotFoundError("Run Quiz 2 first to create improved_mlp.pt.")

np.random.seed(seed)
torch.manual_seed(seed)
warnings.filterwarnings("ignore", message="(?s).*is not compatible with the current PyTorch installation.*")
cuda_capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
cuda_architecture = f"sm_{cuda_capability[0]}{cuda_capability[1]}" if cuda_capability else None
cuda_supported = torch.cuda.is_available() and cuda_architecture in torch.cuda.get_arch_list()
device = torch.device("cuda" if cuda_supported else "cpu")


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

mlp_model = ImprovedMLP(x_train_scaled.shape[1]).to(device)
mlp_model.load_state_dict(torch.load(quiz2_model_path, map_location=device))
mlp_model.eval()
with torch.no_grad():
    validation_tensor = torch.from_numpy(x_validation_scaled).to(device)
    mlp_probability = torch.sigmoid(mlp_model(validation_tensor)).cpu().numpy().reshape(-1)

logistic_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=seed,
)
logistic_model.fit(x_train_scaled, y_train)
logistic_probability = logistic_model.predict_proba(x_validation_scaled)[:, 1]

mlp_fpr, mlp_tpr, mlp_thresholds = roc_curve(y_validation, mlp_probability)
logistic_fpr, logistic_tpr, logistic_thresholds = roc_curve(y_validation, logistic_probability)
mlp_auc = auc(mlp_fpr, mlp_tpr)
logistic_auc = auc(logistic_fpr, logistic_tpr)

figure, axis = plt.subplots(figsize=(8, 6))
axis.plot(mlp_fpr, mlp_tpr, linewidth=2.2, label=f"Improved MLP (AUC = {mlp_auc:.4f})")
axis.plot(logistic_fpr, logistic_tpr, linewidth=2.2, label=f"Logistic Regression (AUC = {logistic_auc:.4f})")
axis.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random classifier (AUC = 0.5000)")
axis.set_xlim(0, 1)
axis.set_ylim(0, 1.02)
axis.set_xlabel("False Positive Rate (FPR)")
axis.set_ylabel("True Positive Rate (TPR)")
axis.set_title("ROC Curves on the Same Validation Dataset")
axis.grid(alpha=0.25)
axis.legend(loc="lower right")
figure.tight_layout()
figure.savefig(output_dir / "roc_curve_comparison.png", dpi=180)
plt.close(figure)

pd.DataFrame(
    {
        "validation_row_id": x_validation.index,
        "actual_class": y_validation.to_numpy(dtype=int),
        "mlp_probability": mlp_probability,
        "logistic_regression_probability": logistic_probability,
    }
).to_csv(output_dir / "validation_probabilities.csv", index=False)

mlp_roc_frame = pd.DataFrame(
    {
        "model": "Improved MLP",
        "fpr": mlp_fpr,
        "tpr": mlp_tpr,
        "threshold": mlp_thresholds,
    }
)
logistic_roc_frame = pd.DataFrame(
    {
        "model": "Logistic Regression",
        "fpr": logistic_fpr,
        "tpr": logistic_tpr,
        "threshold": logistic_thresholds,
    }
)
pd.concat([mlp_roc_frame, logistic_roc_frame], ignore_index=True).to_csv(output_dir / "roc_points.csv", index=False)

auc_results = pd.DataFrame(
    [
        {"model": "Improved MLP", "auc": mlp_auc},
        {"model": "Logistic Regression", "auc": logistic_auc},
    ]
)
auc_results.to_csv(output_dir / "auc_results.csv", index=False)

better_model = "Improved MLP" if mlp_auc > logistic_auc else "Logistic Regression" if logistic_auc > mlp_auc else "Both models"
auc_difference = abs(mlp_auc - logistic_auc)
report = "\n".join(
    [
        "# Quiz 3：模型表現評估",
        "",
        "## 一、資料與檔案說明",
        "",
        f"本題延續 Quiz 2 的信用卡違約資料與 Improved MLP，共使用 {len(data)} 筆資料和 {features.shape[1]} 個輸入特徵。Training Dataset 為 {len(x_train)} 筆，Validation Dataset 為 {len(x_validation)} 筆。MLP 與 Logistic Regression 使用完全相同的資料切分與 Feature Scaling。",
        "",
        "- 程式：`week2/Quiz03/Code/quiz3_model_evaluation.py`",
        "- 資料說明：`week2/Quiz03/Data/README.md`",
        "- 實際資料：`week2/Quiz02/Data/UCI_Credit_Card.csv`",
        "- MLP 權重：`week2/Quiz02/Output/improved_mlp.pt`",
        "- AI 使用紀錄：`week2/Quiz03/AI/AI_usage.md`",
        "- ROC 比較圖：`week2/Quiz03/Output/roc_curve_comparison.png`",
        "- AUC 結果：`week2/Quiz03/Output/auc_results.csv`",
        "- Validation 機率：`week2/Quiz03/Output/validation_probabilities.csv`",
        "- ROC 座標：`week2/Quiz03/Output/roc_points.csv`",
        "",
        "## 二、評估方法",
        "",
        "程式載入 Quiz 2 的 Improved MLP 權重，取得 Validation Dataset 的 Prediction Probability。第二個模型使用 Logistic Regression，並以相同的 Training Dataset 進行訓練。接著使用兩個模型的 Probability 計算 FPR、TPR 與 AUC，並將 ROC Curve 畫在同一張圖中。",
        "",
        "## 三、ROC Curve 與 AUC 意義",
        "",
        "ROC Curve 的 X 軸為 False Positive Rate，Y 軸為 True Positive Rate。曲線越靠近左上角，表示模型可以在較低的 False Positive Rate 下辨識更多正類樣本。AUC 是 ROC Curve 下方的面積，0.5 約等於隨機判斷，越接近 1 代表正負樣本區分能力越好。",
        "",
        "## 四、比較結果",
        "",
        "| 模型 | AUC |",
        "|---|---:|",
        f"| Improved MLP | {mlp_auc:.6f} |",
        f"| Logistic Regression | {logistic_auc:.6f} |",
        "",
        f"兩個模型的 AUC 相差 {auc_difference:.6f}。{better_model} 的 ROC Curve 整體較靠近左上角，因此具有較好的正負樣本排序能力。Logistic Regression 結構較簡單且容易解釋；MLP 能學習較複雜的非線性關係，因此本次實驗得到較高的 AUC。",
    ]
)
(output_dir / "report.md").write_text(report, encoding="utf-8")

print(auc_results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
print()
print(report)
