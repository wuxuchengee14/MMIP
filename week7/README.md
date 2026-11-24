# Week 7：LLM／VLM 部署、評估與偏好微調

延續其他週的 quiz1～quiz3 / AI / Code / Data / Output 結構。三題配分 30＋20＋40，共 90 分；完成度代表每個要求已有實驗證據，不保證老師評分，也不代表每項模型能力都改善。

| 題目 | 內容 | 完整報告 |
|---|---|---|
| Quiz 1（30 分） | Qwen3-8B；20 題；BF16／NF4 × 基本／優化提示；客觀與商用評分；平均 GPU 記憶體 | [report.md](quiz1/Output/report.md) |
| Quiz 2（20 分） | Qwen2.5-VL-7B 對 SmolVLM2-2.2B；20 道圖像題；相同圖與提示；客觀與商用看圖評分 | [report.md](quiz2/Output/report.md) |
| Quiz 3（40 分） | 自建偏好資料；LLM／VLM DPO＋LoRA；獨立目標與通用測試；五種影像條件 | [report.md](quiz3/Output/report.md) |

主要結果見 [REPORT.md](REPORT.md)，逐項配分見 [RUBRIC.md](RUBRIC.md)，提交與排除項目見 [SUBMISSION.md](SUBMISSION.md)。

## 環境與執行

在儲存庫根目錄，新環境：

```powershell
conda env create -f week7/environment.yml
conda activate MMIP_LLM
```

已有環境則執行 `python -m pip install -r week7/requirements.txt`。版本沿用 RTX 5090 已執行成功的 torch 2.9.1+cu128／transformers 4.57.6；Windows 不需要 FlashAttention。

- Quiz 1：[全部步驟與四組指令](quiz1/ALL_TASKS.md)。Qwen3 固定 revision；使用者以商用模型手動審閱全部二十題、四組匿名答案。
- Quiz 2：[操作表](quiz2/README.md)；Git Bash 執行 `bash week7/quiz2/run_quiz2.sh`。商用審核每題須真的上傳原圖。
- Quiz 3：[操作表](quiz3/README.md)；PowerShell 執行 `& ./week7/quiz3/run_quiz3.ps1`，或 Git Bash 的 `run_quiz3.sh`。此題不需要商用 API。
- 三個基底模型不在 GitHub。可使用 `python week7/download_models.py` 下載固定版本至各題 models/ 快取。
- GitHub 保存已完成紀錄；正式程式不會直接覆寫既有輸出。需要重做時先將該次 Output 備份另存。Quiz 3 放回 adapter 或重新訓練後，才能重做微調後推論及含權重的完整檢核。

只核對 GitHub 已提交的資料、原始結果與檔案雜湊、不下載模型或改寫結果：

```powershell
python week7/verify_saved_results.py
```

## GitHub 內容

依照其他週，保留程式、環境、AI 使用紀錄、資料來源、報告、圖表、CSV／JSON 結果。Week 7 的自建題庫與控制圖很小，另保留這些 Data 檔案，方便直接審閱；完整 Flickr8k、模型快取、adapter safetensors、checkpoint、tokenizer 複本、log、smoke 與提交 zip 不上傳。本機原始實驗未刪除或覆寫。

原始結果 JSON 中的本機路徑是當時的歷史 metadata，保留原始 bytes 以維持既有雜湊；唯讀核對程式使用儲存庫相對路徑。網頁報告的圖片連結使用相對路徑。
