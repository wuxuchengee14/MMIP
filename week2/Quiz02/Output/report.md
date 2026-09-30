# Quiz 2：深度學習信用卡違約預測

## 一、資料與檔案說明

本題使用 Kaggle 的 Default of Credit Card Clients Dataset，共有 30000 筆資料及 23 個輸入特徵。資料依照 80% 與 20% 分為 Training Dataset 和 Validation Dataset，分別為 24000 筆與 6000 筆。違約類別的比例為 0.2212。

- 程式：`week2/Quiz02/Code/quiz2_credit_default.py`
- 原始資料：`week2/Quiz02/Data/UCI_Credit_Card.csv`
- Kaggle 壓縮檔：`week2/Quiz02/Data/credit-card-default.zip`
- 套件清單：`week2/Quiz02/requirements.txt`
- AI 使用紀錄：`week2/Quiz02/AI/AI_usage.md`
- Loss 曲線：`week2/Quiz02/Output/loss_curves.png`
- 指標比較圖：`week2/Quiz02/Output/metric_comparison.png`
- 混淆矩陣：`week2/Quiz02/Output/confusion_matrices.png`
- 單筆預測：`week2/Quiz02/Output/sample_prediction.csv`
- 模型權重：`week2/Quiz02/Output/baseline_mlp.pt`、`week2/Quiz02/Output/improved_mlp.pt`

## 二、資料處理與基準模型

我先移除 ID 欄位，再使用 StandardScaler 標準化輸入特徵。基準 MLP 使用 60 Epochs、Batch Size 4096、Learning Rate 0.001、Adam Optimizer，以及 64、32 個神經元的隱藏層。由於資料類別不平衡，Loss Function 使用加權 Binary Cross Entropy。

## 三、改善策略

改善版 MLP 的隱藏層為 128、64、32 個神經元，加入 Batch Normalization、Dropout、L2 Regularization 和 ReduceLROnPlateau，Optimizer 改為 AdamW。Dropout Rate 為 0.30 與 0.20，Weight Decay 為 0.0005。

## 四、結果與觀察

| 模型 | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Baseline MLP | 0.7588 | 0.4650 | 0.6014 | 0.5245 |
| Improved MLP | 0.7525 | 0.4555 | 0.6089 | 0.5211 |

基準模型的最佳 Validation Loss 為 0.888748，改善版為 0.877062。基準模型最後的 Loss Gap 為 0.030851，改善版為 0.019932。

改善版縮小了 Training Loss 與 Validation Loss 的差距，顯示抑制 overfitting 的效果較好。
改善版取得較低的最佳 Validation Loss。
改善版最後 10 個 Epoch 的 Validation Loss 較穩定。
改善版沒有提高 Validation F1-Score。

程式展示的單筆 Validation Sample 為資料列 6907，實際類別為 0，預測類別為 0，預測違約機率為 0.336543。完整資料保存在 `week2/Quiz02/Output/sample_prediction.csv`。