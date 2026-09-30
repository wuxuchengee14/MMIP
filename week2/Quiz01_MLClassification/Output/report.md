# Quiz 1：機器學習分類任務

## 一、資料與檔案說明

本題使用 Scikit-Learn 產生二元分類模擬資料，共有 1200 筆資料及 6 個輸入特徵。資料依照 75% 與 25% 分為 Training Dataset 和 Validation Dataset，分別為 900 筆與 300 筆。

此資料集由 `make_classification` 產生，其中 5 個特徵包含與分類有關的資訊，另有 1 個特徵由其他特徵線性組合而成。產生資料時會打亂特徵順序，因此不能直接指定哪一欄是 redundant feature。所有輸入欄位都是連續數值，沒有對應真實世界的物理單位。

| 欄位 | 型態 | 說明 |
|---|---|---|
| `feature_1` | 浮點數 | 第一個合成數值特徵，作為模型分類依據之一 |
| `feature_2` | 浮點數 | 第二個合成數值特徵，作為模型分類依據之一 |
| `feature_3` | 浮點數 | 第三個合成數值特徵，作為模型分類依據之一 |
| `feature_4` | 浮點數 | 第四個合成數值特徵，作為模型分類依據之一 |
| `feature_5` | 浮點數 | 第五個合成數值特徵，作為模型分類依據之一 |
| `feature_6` | 浮點數 | 第六個合成數值特徵，作為模型分類依據之一 |
| `target` | 整數 | 二元分類標籤；0 代表負類，1 代表正類 |

資料產生時將類別比例設定為約 68% 的類別 0 與 32% 的類別 1，並加入少量標籤雜訊，使分類任務更接近真實資料。模型訓練時，`feature_1` 到 `feature_6` 為 X，`target` 為 y；`target` 不會參與 Feature Scaling。

- 程式：`week2/Quiz01_MLClassification/Code/quiz1_classification.py`
- 資料：`week2/Quiz01_MLClassification/Data/classification_data.csv`
- AI 使用紀錄：`week2/Quiz01_MLClassification/AI/AI_usage.md`
- 評估結果：`week2/Quiz01_MLClassification/Output/metrics.csv`
- 混淆矩陣：`week2/Quiz01_MLClassification/Output/confusion_matrices.png`
- Threshold 曲線：`week2/Quiz01_MLClassification/Output/threshold_curves.png`
- 模型比較圖：`week2/Quiz01_MLClassification/Output/model_comparison.png`

## 二、資料前處理與模型

我使用 StandardScaler 對特徵進行標準化。第一個模型為 Logistic Regression，初始 Threshold 為 0.50，根據 Validation Dataset 的 F1-Score 微調後選擇 0.40。第二個模型使用 Random Forest，選出的 Threshold 為 0.28。

## 三、結果

| 模型 | Threshold | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|---:|
| Logistic Regression 初始 | 0.50 | 0.8000 | 0.7937 | 0.5155 | 0.6250 |
| Logistic Regression 微調 | 0.40 | 0.8133 | 0.7356 | 0.6598 | 0.6957 |
| Random Forest 微調 | 0.28 | 0.9033 | 0.8148 | 0.9072 | 0.8585 |

Logistic Regression 降低 Threshold 後，Recall 與 F1-Score 提升，但 Precision 稍微下降。微調後共有 23 筆 False Positive 和 33 筆 False Negative，主要錯誤為 False Negative。

Random Forest 有 20 筆 False Positive 和 9 筆 False Negative，主要錯誤為 False Positive。整體而言，Random Forest 的 Accuracy、Precision、Recall 與 F1-Score 都較高，分類錯誤也較少。