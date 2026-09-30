# Quiz 3：模型表現評估

## 一、資料與檔案說明

本題延續 Quiz 2 的信用卡違約資料與 Improved MLP，共使用 30000 筆資料和 23 個輸入特徵。Training Dataset 為 24000 筆，Validation Dataset 為 6000 筆。MLP 與 Logistic Regression 使用完全相同的資料切分與 Feature Scaling。

- 程式：`week2/Quiz03/Code/quiz3_model_evaluation.py`
- 資料說明：`week2/Quiz03/Data/README.md`
- 實際資料：`week2/Quiz02/Data/UCI_Credit_Card.csv`
- MLP 權重：`week2/Quiz02/Output/improved_mlp.pt`
- AI 使用紀錄：`week2/Quiz03/AI/AI_usage.md`
- ROC 比較圖：`week2/Quiz03/Output/roc_curve_comparison.png`
- AUC 結果：`week2/Quiz03/Output/auc_results.csv`
- Validation 機率：`week2/Quiz03/Output/validation_probabilities.csv`
- ROC 座標：`week2/Quiz03/Output/roc_points.csv`

## 二、評估方法

程式載入 Quiz 2 的 Improved MLP 權重，取得 Validation Dataset 的 Prediction Probability。第二個模型使用 Logistic Regression，並以相同的 Training Dataset 進行訓練。接著使用兩個模型的 Probability 計算 FPR、TPR 與 AUC，並將 ROC Curve 畫在同一張圖中。

## 三、ROC Curve 與 AUC 意義

ROC Curve 的 X 軸為 False Positive Rate，Y 軸為 True Positive Rate。曲線越靠近左上角，表示模型可以在較低的 False Positive Rate 下辨識更多正類樣本。AUC 是 ROC Curve 下方的面積，0.5 約等於隨機判斷，越接近 1 代表正負樣本區分能力越好。

## 四、比較結果

| 模型 | AUC |
|---|---:|
| Improved MLP | 0.774530 |
| Logistic Regression | 0.708080 |

兩個模型的 AUC 相差 0.066450。Improved MLP 的 ROC Curve 整體較靠近左上角，因此具有較好的正負樣本排序能力。Logistic Regression 結構較簡單且容易解釋；MLP 能學習較複雜的非線性關係，因此本次實驗得到較高的 AUC。