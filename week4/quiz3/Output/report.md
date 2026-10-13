# Quiz 3：ViT 與 ResNet 微調

## 方法

ViT-B/16 與 ResNet-50 使用 ImageNet-1K V1 預訓練權重，分類層改為 11 類；先訓練 head，再以較小 learning rate 微調全模型。使用相同官方 train/val/test、RGB 三通道、224×224 與 ImageNet 正規化，不使用隨機擴增。AdamW、AMP、梯度裁切、驗證 Accuracy 選權重及 early stopping。

## 測試結果

| model | top1 | top5 | macro_f1 | macro_auc | parameters | training_seconds | test_seconds | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vit_b_16 | 0.966419 | 0.999213 | 0.965712 | 0.998868 | 85807115 | 717.416873 | 5.935773 | 12 |
| resnet50 | 0.957982 | 0.999213 | 0.953833 | 0.998569 | 23530571 | 205.959535 | 3.021277 | 5 |

本次 Macro-AUC 較高者為 vit_b_16。Macro-AUC 是各器官 one-vs-rest ROC-AUC 的算術平均，輸入為 softmax 機率。Accuracy 與 AUC 衡量面向不同，應同時閱讀混淆矩陣與各類 ROC。

## 檔案與限制

各模型目錄保存權重、完整逐 epoch 記錄、ROC、各類 AUC、混淆矩陣、逐筆機率與 prediction_examples.png。放大 28×28 CT 不會增加細節；單一 seed 比較不宣稱統計顯著性。training_seconds 含 head warmup，test_seconds 包含資料載入與傳輸。
