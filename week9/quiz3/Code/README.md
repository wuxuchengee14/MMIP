# Week 9 共用程式

`app.py` 提供 Flask 聊天室。分流規格、OCR、會話狀態與工單政策在 `triage/`。實驗 A–D 的正式入口是 `scripts/week9_llm.py`，整合評估入口是 `scripts/run_exp3.py`。

請從 Week 9 目錄使用 [PowerShell 操作指南](../../README.md)。訓練及基準資料在 [Quiz 2 Data](../../quiz2/Data/README.md)，圖片與整合資料在 [Quiz 3 Data](../Data/README.md)。正式結果見 [Quiz 2 報告](../../quiz2/Output/report.md) 與 [Quiz 3 報告](../Output/report.md)。

資料產生腳本供追溯原課堂資料。現有 Benchmark 已凍結並記錄雜湊，重現正式比較時直接使用隨附的 JSONL。
