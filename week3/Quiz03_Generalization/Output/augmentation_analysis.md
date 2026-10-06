# 資料擴增前後分析

| model | augmentation | top1 | top5 | macro_auc | total_parameters | trainable_parameters | refit_epochs | lr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plain | False | 0.92086 | 0.99730 | 0.99537 | 304363 | 304363 | 0 | 0.00100 |
| vgg19 | False | 0.92643 | 0.99477 | 0.99378 | 139615307 | 129030155 | 0 | 0.00030 |
| plain | True | 0.92001 | 0.99612 | 0.99535 | 304363 | 304363 | 0 | 0.00100 |
| vgg19 | True | 0.91945 | 0.99505 | 0.99318 | 139615307 | 129030155 | 0 | 0.00030 |

plain：Top-1 下降，差異 -0.084 個百分點；Macro-AUC 差異 -0.00003。結果不保證擴增一定有效。
plain Quiz 2 在各 fold 選定 epoch 的 train−val Top-1 平均差：-0.00667。
plain Quiz 3 在各 fold 選定 epoch 的 train−val Top-1 平均差：-0.02391。
擴增版的訓練分數來自隨機擴增影像，與固定驗證影像難度不同；上述落差僅作輔助觀察。
vgg19：Top-1 下降，差異 -0.697 個百分點；Macro-AUC 差異 -0.00060。結果不保證擴增一定有效。
vgg19 Quiz 2 在各 fold 選定 epoch 的 train−val Top-1 平均差：+0.00320。
vgg19 Quiz 3 在各 fold 選定 epoch 的 train−val Top-1 平均差：-0.00098。
擴增版的訓練分數來自隨機擴增影像，與固定驗證影像難度不同；上述落差僅作輔助觀察。

同 fold 的擴增前後差異：

| model | metric | paired_delta_mean | paired_delta_std |
| --- | --- | --- | --- |
| plain | top1 | -0.00706 | — |
| plain | top5 | 0.00012 | — |
| plain | macro_auc | -0.00024 | — |
| vgg19 | top1 | 0.00049 | — |
| vgg19 | top5 | 0.00012 | — |
| vgg19 | macro_auc | 0.00007 | — |

Kernel 與 XAI 的圖表、逐例分析見 explanations/。
兩套模型同時存在架構、預訓練、通道 normalization 差異，因此準確率差异不能全部歸因於參數量。