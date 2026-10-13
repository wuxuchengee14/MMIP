# 作業要求與成果對照

六個模型均已實際訓練與測試。Gemini 採使用者指定的 10 張案例手動上傳方式，兩模型共 20 筆已匯入。
此表對照作業要求與實際成果；完成項目不代表保證授課者的評分。

| 題目 | 分數 | 要求 | 實際程式／執行後成果 |
| --- | --- | --- | --- |
| Quiz 1 基礎 | 3 | 情感分析資料 | quiz1/Code/prepare.py；Data/imdb/、Output/datasets.json |
| Quiz 1 基礎 | 4 | 影像分類資料 | Data/organamnist/、Output/organ_class_counts.csv、organ_examples.png |
| Quiz 1 基礎 | 3 | 影像與描述資料 | Data/flickr8k/；官方 image-level splits，每圖五句 |
| Quiz 2 基礎 | 5 | RNN 訓練 | quiz2/Code/models.py、train.py；Output/rnn/best.pt、history.csv |
| Quiz 2 基礎 | 5 | 測試分類 | Output/rnn/predictions.csv、metrics.json、examples.json |
| Quiz 2 進階 | 5 | LSTM 訓練 | Output/lstm/best.pt、history.csv、predictions.csv |
| Quiz 2 進階 | 5 | RNN / LSTM 評估比較 | Output/comparison.csv、report.md、各模型 ROC／混淆矩陣 |
| Quiz 3 基礎 | 20 | ViT 微調與分類推論 | quiz3/Code/train.py；Output/vit_b_16/ |
| Quiz 3 進階 | 10 | ResNet 微調與推論 | Output/resnet50/ |
| Quiz 3 進階 | 5 | Macro-AUC 比較 | Output/comparison.csv；各類 ROC／class_auc.csv／metrics.json |
| Quiz 4 基礎 | 15 | 訓練並生成影像描述 | quiz4/Code/models.py、features.py、train.py；Output/lstm/、transformer/ |
| Quiz 4 基礎 | 5 | Gemini 判斷影像符合度 | quiz4/Code/manual_gemini.py；Output/gemini_cases10/ 的 10 張原圖、兩批提示詞／原始回覆、20 筆評分與比較報告 |
| Quiz 4 進階 | 5 | BLEU 或 BERTScore | 本專案選 BLEU-1～4；各模型 greedy/beam/bleu.json |
| Quiz 4 進階 | 10 | 方法的回覆時間與表現 | 兩個 decoder × greedy/beam；comparison.csv、timing.json、encoder_timing.json |
| 合計 | 100 | 全部基礎與進階 | 各題 Output/report.md、根目錄 REPORT.md、completion.json |

完整流程：bash week4/train_rtx5090.sh。
本次 Gemini 使用網頁手動上傳，沒有宣稱原 100 張／200 筆 API 方案已完成。
人工挑選的 10 張案例僅供展示，不作為全體測試集正確率的估計。
