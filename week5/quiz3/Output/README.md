# 正式描述與跨週比較

caption 階段產生 clip_transformer/ 的權重、曲線與 greedy/beam 預測、BLEU、時間及案例。
compare 階段從實際 Week4/Week5 輸出重算六組 BLEU/BERTScore，保留 scorer hash 與逐圖 P/R/F1。
comparison.json/csv、report.md 為最終對照。沒有呼叫 Gemini 或外部生成 API。
