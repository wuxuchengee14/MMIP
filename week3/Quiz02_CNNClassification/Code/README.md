# Quiz 2：訓練 CNN 影像分類模型

主要執行程式為 `run.py`，展示 notebook 為 `quiz2.ipynb`。資料沿用 Quiz 1 的 OrganAMNIST 與固定切分。

- 報告：[`../Output/report.md`](../Output/report.md)
- 模型比較：[`../Output/model_comparison.csv`](../Output/model_comparison.csv)
- Plain CNN 權重：[`../Output/plain/selected/best.pt`](../Output/plain/selected/best.pt)
- Plain CNN 測試指標：[`../Output/plain/test/metrics.json`](../Output/plain/test/metrics.json)
- VGG-19 權重：[`../Output/vgg19/selected/best.pt`](../Output/vgg19/selected/best.pt)
- VGG-19 測試指標：[`../Output/vgg19/test/metrics.json`](../Output/vgg19/test/metrics.json)

每個模型的 `test` 資料夾包含 Top-1／Top-5 Accuracy、Macro-AUC、各類別 ROC、混淆矩陣、分類報告與逐筆預測結果。
