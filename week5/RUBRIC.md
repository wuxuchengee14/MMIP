# Week5 要求對照

三張投影片合計 95 分，未自行新增其餘 5 分要求。下表列實作與預期成果，不代表尚未執行的訓練已完成。

| 題目 | 分數 | 要求 | 程式／成果 |
| --- | --- | --- | --- |
| Quiz1 基礎 | 5 | 準備檢索資料 | quiz1/Code/prepare.py；Data/cub/manifest.json、retrieval_splits.json |
| Quiz1 基礎 | 10 | 建立與訓練檢索模型 | models.py、train.py；SupCon+CE、history.csv、best.pt |
| Quiz1 基礎 | 5 | 檢索推論與展示 | retrieval_results.csv、metrics.json、retrieval_examples.png |
| Quiz1 進階 | 5 | CNN / ViT Backbone | Output/resnet50/、vit_b_16/ |
| Quiz1 進階 | 5 | 檢索比較 | comparison.csv/json、report.md |
| Quiz2 基礎 | 5 | 影像／文字多模態資料 | 共用 CUB，原始十句人工描述按圖片切分 |
| Quiz2 基礎 | 10 | 影像／文字 Backbone 融合分類 | ImageBackbone + frozen DistilBERT + projection/concat/MLP |
| Quiz2 基礎 | 5 | 訓練與多模態推論 | best.pt、curves、predictions.csv、prediction_examples.png |
| Quiz2 進階 | 10 | 更換至少一種 Backbone | ResNet50 換 ViT-B/16，文字與融合設定保持相同 |
| Quiz2 進階 | 5 | 組合比較 | comparison.csv/json、report.md、Macro-F1、混淆矩陣 |
| Quiz3 基礎 | 5 | Caption 資料準備 | Week4 Flickr8k 唯讀重用；flickr_manifest.json |
| Quiz3 基礎 | 10 | CLIP Image Encoder + Decoder | frozen CLIP ViT-B/32 + trainable adapter + Transformer |
| Quiz3 基礎 | 5 | 訓練與描述生成 | best.pt、history.csv、greedy/beam predictions、範例圖 |
| Quiz3 進階 | 10 | Week4 與本週 BLEU、BERTScore | compare.py；同 1000 張與五參考句，六組比較、scorer hash、逐圖 P/R/F1 |

實際完成狀態以 completion.json 與正式輸出為準；單元測試／GPU 隨機輸入檢查不計為正式訓練成果。
