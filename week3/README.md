# Week 3：CNN 影像分類

Week 3 使用 OrganAMNIST 腹部 CT 灰階影像完成資料準備、CNN 分類、泛化能力提升與模型解釋。每一題均依照 `AI / Code / Data / Output` 分類，完整報告固定放在各題的 `Output/report.md`。

```text
week3/
├── Quiz01_DatasetPreparation/
│   ├── AI/
│   ├── Code/
│   ├── Data/
│   └── Output/
├── Quiz02_CNNClassification/
│   ├── AI/
│   ├── Code/
│   ├── Data/
│   └── Output/
└── Quiz03_Generalization/
    ├── AI/
    ├── Code/
    ├── Data/
    └── Output/
```

- Quiz 1：說明資料來源、11 個器官類別、資料切分與分類問題。
- Quiz 2：比較自行設計的 Plain CNN 與 ImageNet 預訓練 VGG-19，包含超參數實驗、Top-1／Top-5 Accuracy、ROC、Macro-AUC 與參數量。
- Quiz 3：加入 Data Augmentation，比較泛化表現，並以 Kernel 視覺化和 Grad-CAM 解釋模型。
