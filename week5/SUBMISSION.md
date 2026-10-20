# Week5 繳交檔案對照

以下是依投影片整理的成果，仍應以老師指定的繳交格式為準。

| 題目 | 報告與展示 | 程式 | 資料說明 |
| --- | --- | --- | --- |
| Quiz1 | quiz1/Output/report.md、comparison.csv、comparison.png；兩模型的 learning_curves.png、retrieval_examples.png、metrics.json、retrieval_results.csv | quiz1/Code/ 全部程式與 quiz1.ipynb | quiz1/Data/README.md、cub/manifest.json、classes.json、retrieval_splits.json；Output/data_audit.json |
| Quiz2 | quiz2/Output/report.md、comparison.csv、comparison.png；兩模型的 learning_curves.png、prediction_examples.png、metrics.json、confusion_matrix.png、predictions.csv | quiz2/Code/ 全部程式與 quiz2.ipynb | quiz2/Data/README.md；共用 Quiz1 資料 |
| Quiz3 | quiz3/Output/report.md、comparison.csv、comparison.png；clip_transformer/learning_curves.png、greedy/beam 的 caption_examples.png、predictions.json、bleu.json、timing.json；bertscore/ 六組評估與逐圖分數 | quiz3/Code/ 全部程式與 quiz3.ipynb | quiz3/Data/README.md、flickr_manifest.json |

共同保留：common/、run_all.py、predict.py、設定 JSON、environment.yml、requirements.txt、兩個啟動腳本、README.md、DATA_SOURCES.md、RUBRIC.md、VALIDATION.md、REPORT.md、completion.json，以及各題 AI/。

若需重現推論，另外保留每模型 best.pt、complete.json，以及 Quiz1 embeddings.npz、Quiz2 text_features/complete.json、Quiz3 clip_features/complete.json 與 Week4 詞彙表。
若需續訓，還要 last.pt 與原始資料／快取。完整重現需要 Week4 Flickr8k 及四組描述預測，詳見 README.md。

原始 CUB 壓縮檔、影像與大型特徵快取會占用空間；不要只為上傳大小而刪除本機原始成果。
可另複製繳交資料夾，依老師要求決定是否附權重與原始資料。completion.json 為 true 才代表本流程已產生全部要求的實際成果。

## GitHub 版本

本儲存庫的路徑為 `week5/quiz1～quiz3/`，每題均包含 `AI / Code / Data / Output`，報告位於 `Output/report.md`。原始資料、模型權重、向量／特徵快取、訓練日誌和硬體資訊均依根目錄 `.gitignore` 排除；圖表、逐筆預測、指標與報告會上傳。上述被忽略的檔案仍保留在本機，GitHub 版本不包含完整離線推論或續訓所需的全部檔案。
