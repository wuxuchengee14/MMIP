# Quiz 1：準備資料集

主要程式為 `prepare.py`，展示 notebook 為 `quiz1.ipynb`。

- 報告：[`../Output/report.md`](../Output/report.md)
- 原始資料：[`../Data/organamnist.npz`](../Data/organamnist.npz)
- 固定 fold：[`../Data/folds.npz`](../Data/folds.npz)
- 切分資訊：[`../Data/split_manifest.json`](../Data/split_manifest.json)
- 類別數量：[`../Output/official_class_counts.csv`](../Output/official_class_counts.csv)
- 類別分布圖：[`../Output/class_distribution.png`](../Output/class_distribution.png)
- 影像範例：[`../Output/class_examples.png`](../Output/class_examples.png)
- Fold 分布：[`../Output/fold_class_counts.csv`](../Output/fold_class_counts.csv)

資料使用 OrganAMNIST 的官方 train、validation 與 test split；開發資料另外保存固定的 stratified fold 索引。
