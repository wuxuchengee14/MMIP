# 繳交檔案導覽

先閱讀 [作業繳交報告](SUBMISSION_REPORT.md)，再依 [要求對照](RUBRIC.md) 查閱各題 Code/ 與 Output/。

- Quiz 1：資料來源、切分、類別與影像／文字樣本。
- Quiz 2：RNN/LSTM 原始碼、曲線、測試預測、Accuracy/F1/ROC-AUC 比較。
- Quiz 3：ViT/ResNet 微調、測試預測、各類 ROC 與 Macro-AUC。
- Quiz 4：兩種 caption decoder、greedy/beam、BLEU、時間及十張 Gemini 手動案例。

[Gemini 案例報告](quiz4/Output/gemini_cases10/report.md) 附原始圖片、提示詞與回覆，介面模型標記由使用者紀錄；本批為人工挑選案例，不是隨機抽樣效能估計。

模型權重另附 models ZIP，六個 best.pt 可依相同路徑合併本包。大型原始資料、特徵快取與 optimizer 續訓檔未包含，重新執行須先下載／重建資料。

完整性：FILE_MANIFEST.json 列出包內檔案大小與 SHA-256（清單自身除外）。
