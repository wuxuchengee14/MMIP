# Quiz 3 實驗報告

所有數字來自實際輸出。Accuracy 使用 0～1 比例；CV 標準差採樣本標準差。
交叉驗證用於設定選擇，選出設定的 CV 分數不是無偏的最終泛化估計；最終評估使用官方 test。

本階段只執行第一折，但完成基礎與進階項目。各 LR 在同一驗證集比較；沒有五折平均或可計算的 fold 標準差。
使用選定設定的最佳驗證 checkpoint 測試，不另做開發池重訓；表中的 refit_epochs=0。

## 擴增與比較設計

訓練時使用 RandomAffine 與灰階亮度／對比擾動；不做左右翻轉。具體幅度見 config.json。
Validation / test 無隨機擴增。沿用 Quiz 2 的 LR、fold、初始化 seed、optimizer 與 epoch 上限。
單折使用相同 epoch 上限與 early stopping 規則，實際停止 epoch 可不同。

## plain

| lr | top1_mean | top1_std | top5_mean | top5_std | macro_auc_mean | macro_auc_std |
| --- | --- | --- | --- | --- | --- | --- |
| 0.00100 | 0.97649 | — | 0.99976 | — | 0.99950 | — |

選用 learning rate：0.001。

| model | augmentation | top1 | top5 | macro_auc | total_parameters | trainable_parameters | refit_epochs | lr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plain | True | 0.92001 | 0.99612 | 0.99535 | 304363 | 304363 | 0 | 0.00100 |

各類 ROC、混淆矩陣、分類報告與逐筆預測位於該模型的 test 資料夾。

測試召回率最低的類別：kidney-left (0.769)、heart (0.824)、kidney-right (0.858)。
最多的單向混淆：kidney-right → kidney-left，共 171 張；可搭配錯誤影像與 Grad-CAM 分析。

## vgg19

| lr | top1_mean | top1_std | top5_mean | top5_std | macro_auc_mean | macro_auc_std |
| --- | --- | --- | --- | --- | --- | --- |
| 0.00030 | 0.99306 | — | 0.99976 | — | 0.99994 | — |

選用 learning rate：0.0003。

| model | augmentation | top1 | top5 | macro_auc | total_parameters | trainable_parameters | refit_epochs | lr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vgg19 | True | 0.91945 | 0.99505 | 0.99318 | 139615307 | 129030155 | 0 | 0.00030 |

各類 ROC、混淆矩陣、分類報告與逐筆預測位於該模型的 test 資料夾。

測試召回率最低的類別：kidney-left (0.722)、kidney-right (0.858)、femur-left (0.872)。
最多的單向混淆：kidney-left → spleen，共 235 張；可搭配錯誤影像與 Grad-CAM 分析。
