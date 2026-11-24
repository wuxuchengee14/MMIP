# Week 9 多模態多輪客服分流

EriXNet 是 3C 商城客服示範。三題共用一套工單規格與可執行的聊天室，各題的資料與實驗證據按評分要求整理。

| 項目 | 配分 | 成果 |
|---|---:|---|
| 作業一 多模態多輪對話機器人 | 15 | [業務情境、下游任務、規格與架構](quiz1/Output/report.md) |
| 作業二 建立與比較分流節點 | 50 | [四種方法的實測與比較](quiz2/Output/report.md) |
| 作業三 整合與優化 | 35 | [聊天室、UX、離題與防禦結果](quiz3/Output/report.md) |

三題整理自提供的 `hw_project.zip` 與原 Word 報告。補跑的實驗與修正記在上列報告，當中的表格與圖以本次執行的 JSON 為準。

[下載完整 Word 報告](Week9_作業報告.docx) · [查看繳交內容與證據](SUBMISSION.md)

## 在 Windows 重現

在 PowerShell 切換至本資料夾。RTX 5090 可沿用 `MMIP_LLM`。若需重建 PyTorch，先安裝 CUDA 12.8 版 PyTorch，再安裝本作業套件。

```powershell
conda activate MMIP_LLM
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

圖片辨識需要 Tesseract 與 `chi_tra`、`eng` 語言檔。可使用下方指令安裝到作業目錄，也可沿用已在系統 PATH 的 Tesseract。

```powershell
conda create --yes --prefix ./.runtime/ocr --override-channels -c conda-forge tesseract
```

| 步驟 | 用途 | 指令 |
|---|---|---|
| 1 | 建立 CPU 聊天室使用的路由權重 | `./run_week9.ps1 -Stage prepare` |
| 2 | 啟動聊天室 | `./run_week9.ps1 -Stage app` |
| 3 | 跑 Quiz 3 對照實驗 | `./run_week9.ps1 -Stage integration` |
| 4 | 執行單元及整合測試 | `./run_week9.ps1 -Stage test` |
| 5 | 重新執行 Quiz 2 LoRA SFT | `./run_week9.ps1 -Stage train` |
| 6 | 重新比較 A–D | `./run_week9.ps1 -Stage benchmark` |
| 7 | 從原始結果更新 Markdown 報告和比較圖 | `./run_week9.ps1 -Stage report` |

聊天室預設在 `http://127.0.0.1:5000`。可輸入「我想退貨」、接著問「今天天氣怎麼樣」，再上傳 `quiz3/Data/images/order_ship.png` 看到工單。`/demo/1` 與 `/demo/2` 是預置範例。第 2 步會佔用終端機，其他步驟可在另一個終端執行。

重跑 Quiz 2 時可在步驟 5 訓練 LoRA adapter，或從本機繳交檔放回 `quiz2/Output/adapter`。首次重跑會下載固定版本 Qwen3 到 `.runtime`，推論使用本機 GPU。第一次啟動 Quiz 3 示範前，執行第 1 步建立 System One 與 GRU 權重。

共用工單規格、程式與正式實驗入口位於 `quiz3/Code`。Quiz 2 的訓練與基準資料放在 `quiz2/Data`，圖片與整合評估資料放在 `quiz3/Data`，報告與原始結果分別放在各題 `Output`。
