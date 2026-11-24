# Week 7 Quiz 3：LLM／VLM 偏好微調（基礎＋進階 40 分）

此 GitHub 副本已包含實際訓練紀錄、前後測試、評分與圖表。依儲存庫其他週的做法，adapter／checkpoint／tokenizer 檔案不提交；權重雜湊與參數更新證據保留於 training.json。本機完整作業仍保留這些權重。僅審閱已保存結果可使用根目錄的 `verify_saved_results.py`；重跑推論則須重新訓練或放回原 adapter。

本專案依六個配分項目準備資料、真正更新模型 adapter、執行獨立測試及產生報告。以 `Output/completion.json` 與原始證據確認執行完成；「程式寫好」或「短版訓練成功」不算全部完成。實際分數由老師評定，實驗不保證每項指標都提升。

## 配分與實作對應

| 要求 | 配分 | 實作與證據 |
|---|---:|---|
| 自訂 LLM 優化方向、至少 100 組 RLHF 用偏好資料 | 5 | 來源忠實 JSON 擷取：200 組 train＋20 組 dev，含 prompt/chosen/rejected、偏好理由與標籤來源 |
| 另外 100 組與優化方向無關的 Benchmark | 5 | `Data/general_100.json`，算術、邏輯、單位、翻譯、科學、地理、資訊、排序、情感、詞義各十題 |
| 實際微調開源 LLM，量化目標及通用能力變化 | 10 | Qwen3-8B，NF4＋LoRA＋DPO，前後各跑 100 題目標＋100 題通用測試 |
| 自訂 VLM 優化方向、至少 100 組資料 | 10 | 指定顏色＋形狀計數，缺少目標回答 0：200 組 train＋20 組 dev 的圖像偏好資料 |
| 實際微調開源 VLM 並量化改善情形 | 5 | SmolVLM2-2.2B，BF16＋語言注意力 LoRA＋DPO，50 張獨立圖像成對比較 |
| 調整畫質、明暗，觀察平均表現 | 5 | 原圖、JPEG Q15、降至最長邊 64 再放回、亮度 0.5／1.5，共 5 種條件；各 50 題，前後各 250 個輸入 |

「RLHF 用資料」採明確的偏好配對格式。實際使用 [DPO](https://huggingface.co/docs/trl/v0.24.0/en/dpo_trainer) 偏好最佳化；**沒有宣稱執行 PPO、訓練獨立 reward model、蒐集真人評分**。本題圖片沒有要求一定用 PPO，DPO 可直接使用 chosen/rejected 資料實際微調開源模型。LoRA 是更新參數的方法，DPO 是訓練目標，兩者用途不同。

標籤由自建文件 schema、來源值及繪圖物件清單驗證。這是合成、規則標註偏好，不是偽稱的人工 RLHF 標籤。視覺目標是圖像計數與目標缺席判斷；結果不能延伸宣稱自然照片描述幻覺已改善。

## 資料與公平比較

- [資料卡](Data/DATASET_CARD.md)；[100 題通用 Benchmark](Data/general_questions.md)；[五種影像條件預覽](Data/variants_preview.png)。
- 每個模型各 200 組 train、20 組 dev；另外保留 LLM 100 題目標測試、100 題通用測試，VLM 50 張測試圖。train/dev/test 提示與圖像無重複。
- 模型版本沿用 Quiz 1、2 的已下載快取，無需下載另一個大型模型。LLM 的 NF4 基底與 Qwen 非思考模式在前後評估保持一致；VLM 的 BF16、影像縮放與 chat template 也保持一致。
- 兩模型各三個 epoch，LoRA rank=16；只依 dev DPO loss 選最佳 checkpoint，不用 test 挑參數。保持測試集凍結；不得看測試分數後反覆訓練挑高分。
- 相同 prompt、greedy decoding、max_new_tokens=192、相同 GPU。影像變形只用於評估，前後使用同一批檔案。
- 量化指標：JSON 完全正確率、合法格式、schema／欄位正確率、缺失值捏造率、100 題通用正確率、視覺計數正確率／MAE／不存在目標誤報率，以及五條件等權平均。
- VLM 同時保留嚴格整數格式分數與補充的計數內容分數。補充解析只接受單一不歧義數量，不以移除說明的方式修改原本分數；缺席題的不可解析回答也另列，避免誤報率被格式錯誤掩蓋。
- 配對 bootstrap 95% 區間以題目為單位；不能視為多種訓練種子的可信區間。合成資料、小測試集及單次訓練的限制會保留在報告。

## 一次跑完（你目前的 RTX 5090／Windows）

在 `D:/projects/MMIP` 的 PowerShell 執行：

```powershell
conda activate MMIP_LLM
python -m pip install -r week7/quiz3/requirements.txt
& ./week7/quiz3/run_quiz3.ps1
```

Git Bash 也可以：

```bash
conda activate MMIP_LLM
bash week7/quiz3/run_quiz3.sh
```

如果 PowerShell 沒有切到正確 Python，可明確指定：

```powershell
& ./week7/quiz3/run_quiz3.ps1 -PythonExecutable C:/Users/JerryWu/miniconda3/envs/MMIP_LLM/python.exe
```

這題**沒有要求商用模型評估**，所有資料具有可驗證答案，全部使用客觀指標，不需要人工上傳 Gemini／Claude。腳本依序跑基準、微調、微調後測試，最後產生報告及圖表。每次只有一個程序使用 GPU，不同模型權重依序載入。

你目前已完成 Quiz 1／2，因此會直接使用原有快取。若換到另一台電腦，先安裝同版 CUDA 12.8 torch 2.9.1／torchvision 0.24.1 與本 requirements，再執行 `python week7/quiz3/Code/download_models.py` 下載固定版本。此步只補基底模型，不取代 adapter 訓練。

| 步驟 | 工作內容 | 指令（專案根目錄） |
|---|---|---|
| 1 | 建立／驗證凍結資料 | `python week7/quiz3/Code/prepare_data.py` |
| 2 | 檢查版本、CUDA、既有權重與處理器 | `python week7/quiz3/Code/preflight.py` |
| 3 | LLM 微調前，100 目標＋100 通用題 | `python week7/quiz3/Code/evaluate.py --kind llm --stage before` |
| 4 | LLM 正式偏好微調 | `python week7/quiz3/Code/train_dpo.py --kind llm` |
| 5 | LLM 微調後相同兩組測試 | `python week7/quiz3/Code/evaluate.py --kind llm --stage after` |
| 6 | VLM 微調前，50 張圖 × 5 條件 | `python week7/quiz3/Code/evaluate.py --kind vlm --stage before` |
| 7 | VLM 正式偏好微調 | `python week7/quiz3/Code/train_dpo.py --kind vlm` |
| 8 | VLM 微調後相同五條件測試 | `python week7/quiz3/Code/evaluate.py --kind vlm --stage after` |
| 9 | 核對參數更新、計分、產生完整報告 | `python week7/quiz3/Code/score_report.py report` |
| 10 | 產生訓練曲線與前後比較圖 | `python week7/quiz3/Code/plots.py` |
| 11 | 查看是否執行完成 | `python week7/quiz3/Code/score_report.py status` |

完整腳本最後也會產生 `quiz3_submission.zip`，包含資料、程式、最終 adapter、原始結果、報告及圖表，排除基底權重與中間 checkpoint。每個提交檔案的大小與 SHA-256 在 `Output/submission_manifest.json`。需要重新打包可執行 `python week7/quiz3/Code/package_submission.py`。

`run_all.py` 保留已完成結果並核對證據，不覆寫。若有中斷的 JSON 或 trainer 目錄，先查看錯誤並保留該次紀錄，再移到另一個備份目錄重新執行該階段；本版不把半途訓練當完成，也不混接不同設定。開始正式訓練後請保持資料凍結。

程式不估計固定訓練時間；下載已省略，但 Qwen3 的 NF4 載入與 8B 前後推論仍需時間。`Output/*/training.json` 記錄實際秒數與參數變化。低序列長度、短答案與 batch=1 不會讓 GPU 一直滿載。

## 測試與檔案

```powershell
python -m unittest discover -s week7/quiz3/Code -p test_quiz3_tools.py -v
```

可選 `train_dpo.py --kind llm --smoke` 或 `--kind vlm --smoke` 執行兩步實際反向傳播。結果只寫 `Output/smoke/`，不會納入正式完成檢核。離線單元測試中的模擬輸出只在暫存資料夾，不能代替實際模型執行。

繳交時包含：

| 資料夾／檔案 | 內容 |
|---|---|
| `Code/`、README、requirements、兩種執行腳本 | 可重現程式與說明 |
| `Data/` | 全部偏好資料、兩種獨立 Benchmark、圖像與變形、資料卡及 manifest |
| `Output/llm/`、`Output/vlm/` | `training.json`、`before.json`、`after.json`、`adapter/`、`processor/` |
| `Output/metrics.json`、`per_question_scores.json` | 原始答案計分與彙整 |
| `Output/comparison.csv`、`report.md`、`completion.json`、`figures/` | 比較表、實驗報告、40 分要求檢核、訓練曲線與圖表 |

不需要提交基底模型權重、`trainer/` 中間 checkpoint、`Output/smoke/`。以實際結果討論是否改善；若沒有改善或通用能力下降，也應保留，不手動修正模型答案或分數。
