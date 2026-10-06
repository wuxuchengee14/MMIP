# Quiz 2 實驗報告

所有數字來自實際輸出。Accuracy 使用 0～1 比例；CV 標準差採樣本標準差。
交叉驗證用於設定選擇，選出設定的 CV 分數不是無偏的最終泛化估計；最終評估使用官方 test。

本階段只執行第一折，但完成基礎與進階項目。各 LR 在同一驗證集比較；沒有五折平均或可計算的 fold 標準差。
使用選定設定的最佳驗證 checkpoint 測試，不另做開發池重訓；表中的 refit_epochs=0。

## 實驗設計

自行設計 Plain CNN 與 ImageNet 預訓練 VGG-19。僅固定前處理，不做資料擴增。
控制架構、optimizer、batch size 與訓練規則，比較三個 learning rate。
VGG-19 先訓練分類頭，再微調最後卷積區塊；卷積 learning rate 為分類頭的十分之一。

## plain

| lr | top1_mean | top1_std | top5_mean | top5_std | macro_auc_mean | macro_auc_std |
| --- | --- | --- | --- | --- | --- | --- |
| 0.00100 | 0.98356 | — | 0.99963 | — | 0.99974 | — |
| 0.00030 | 0.98198 | — | 0.99963 | — | 0.99970 | — |
| 0.00010 | 0.95384 | — | 0.99915 | — | 0.99831 | — |

選用 learning rate：0.001。

| model | augmentation | top1 | top5 | macro_auc | total_parameters | trainable_parameters | refit_epochs | lr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plain | False | 0.92086 | 0.99730 | 0.99537 | 304363 | 304363 | 0 | 0.00100 |

各類 ROC、混淆矩陣、分類報告與逐筆預測位於該模型的 test 資料夾。

測試召回率最低的類別：kidney-left (0.799)、kidney-right (0.865)、pancreas (0.883)。
最多的單向混淆：kidney-right → kidney-left，共 160 張；可搭配錯誤影像與 Grad-CAM 分析。

## vgg19

| lr | top1_mean | top1_std | top5_mean | top5_std | macro_auc_mean | macro_auc_std |
| --- | --- | --- | --- | --- | --- | --- |
| 0.00100 | 0.98819 | — | 0.99963 | — | 0.99981 | — |
| 0.00030 | 0.99257 | — | 0.99963 | — | 0.99987 | — |
| 0.00010 | 0.98977 | — | 0.99927 | — | 0.99983 | — |

選用 learning rate：0.0003。

| model | augmentation | top1 | top5 | macro_auc | total_parameters | trainable_parameters | refit_epochs | lr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vgg19 | False | 0.92643 | 0.99477 | 0.99378 | 139615307 | 129030155 | 0 | 0.00030 |

各類 ROC、混淆矩陣、分類報告與逐筆預測位於該模型的 test 資料夾。

測試召回率最低的類別：kidney-left (0.855)、femur-left (0.876)、kidney-right (0.882)。
最多的單向混淆：kidney-right → kidney-left，共 156 張；可搭配錯誤影像與 Grad-CAM 分析。
