# Quiz 2：RNN 與 LSTM 情感分析

## 方法

IMDb 正負二分類；單向 RNN / LSTM 共用相同詞彙表、Embedding 維度、hidden 維度、訓練／驗證／測試分割和 early stopping。取最後一個有效 token 的狀態分類，避免 padding 污染；截斷至前 256 tokens（實際長度依設定檔）。交叉熵、AdamW、梯度裁切，驗證 Accuracy 選擇權重，測試不參與挑選。

## 測試比較

| model | accuracy | macro_f1 | roc_auc | parameters | training_seconds | test_seconds | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rnn | 0.756160 | 0.755564 | 0.816201 | 3939330 | 8.543029 | 0.195125 | 5 |
| lstm | 0.833760 | 0.833450 | 0.901322 | 4235778 | 31.591670 | 1.010590 | 14 |

本次測試 Macro-F1 較高者為 lstm。單次固定 seed 的結果不代表統計顯著差異；RNN 可能因長序列記憶不足而落後，應搭配 examples.json 的實際錯誤閱讀，不先假定 LSTM 必定勝出。

## 檔案

各模型子目錄包含 best.pt、last.pt、history.csv、learning_curves.png、metrics.json、classification_report.json、confusion_matrix.png、roc.png、predictions.csv、examples.json 與 timing.json。Top-5 在二分類等於 1，因此本題比較以 Accuracy、F1、正類 ROC-AUC 為主。

設定與資料識別見 complete.json。時間是實際測量，測試時間包含載入與傳輸。
