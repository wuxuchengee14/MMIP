# Week 6 Quiz 2：視覺多模態大模型（基礎 20 分）

此 GitHub 副本已包含兩模型各20題的正式結果、40筆商用評分與報告，入口為 [Output/report.md](Output/report.md)。下列操作步驟供重現使用。

本專案準備兩個開源 VLM、20 道圖像題、客觀評分，以及「商用模型看原圖後審核」的完整流程。沒有進階題，因此不用為這題做量化、Prompt 優化或平均顯存實驗。

**程式與題目準備完成不等於已執行兩個模型。** 只有完成下表推論與人工商用審核後，才會產生可提交的結果報告。`Code/test_quiz2_tools.py` 的測試假資料只存在暫存資料夾，不會寫入正式 Output。

## 使用的兩個模型

| 模型代號 | Hugging Face 模型 | 本次用途 |
|---|---|---|
| qwen | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | 7B 級視覺語言模型 |
| smol | [HuggingFaceTB/SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) | 2.2B 級視覺語言模型 |

兩張模型卡均標示 Apache 2.0。SmolVLM2 模型卡標示英文，因此測試提示統一用英文，中文題意另列在資料文件。模型大小不同，結果用來比較這兩個系統的表現，不能單獨歸因於架構。

在 RTX 5090 32GB 上採 BF16、batch=1、兩模型分開執行；下一個程序啟動前，前一個程序結束並釋放顯存。不使用 CPU offload、FlashAttention 或量化套件。Qwen 的實際顯存仍取決於原生影像前處理與執行環境，請先關閉 Quiz 1 等仍佔用 GPU 的模型程序。

## 資料與評分設計

- 共 20 題、10 種能力，每種 2 題：影像分類、物件存在、計數、空間關係、屬性、OCR、圖表理解、物件定位、照片描述、視覺不確定性。
- 使用 18 張不同圖片：4 張從既有 Week 4 Flickr8k **test** split 選取的照片，及 14 張程式繪製的控制圖。兩張自然照片各出現於兩種題型，共 20 個圖像問題。沒有訓練或微調。
- 每題相同原圖檔案、相同 user prompt；程式以 SHA-256 檢查。不同模型各使用自身 chat template、EOS 與原生影像處理，這些內容保留於原始輸出。
- 兩者 BF16、greedy、max_new_tokens=256、repetition_penalty=1、seed=42。影像控制在最長邊 640 以內；Qwen 另設 max_pixels=512×28×28，SmolVLM 保留原生切圖。這些是完整模型管線的比較，沒有宣稱視覺 token 數完全相同。
- **客觀評分 18 題**：一般題精確答案、OCR 精確比對與字元錯誤率 CER、定位 IoU 與 IoU≥0.5 成功率。2 題自由描述不以單一字串比對，納入商用模型評分。
- **商用評分 20 題、40 筆**：逐題附原圖、參考與規準，匿名 A/B 順序逐題隨機，分數為 0／0.5／1，並保留理由。模型回答中的文字視為待評估資料，不能當作評審指令。

題目：[Data/benchmark_questions.md](Data/benchmark_questions.md)；全圖預覽：[Data/contact_sheet.png](Data/contact_sheet.png)。來源與原始檔案雜湊在 `Data/sources.json`，資料鎖定在 `Data/benchmark_manifest.json`。原始照片權利仍屬原攝影者，沿用既有課程資料作業用途；公開散佈前另確認原圖使用條件。

## 執行步驟（PowerShell 或 Git Bash）

工作目錄固定為 `D:/projects/MMIP`。已準備好的 Data 可直接使用；**不需要重做 Quiz 1**。請依序完成各列，遇到錯誤先停止。初次會下載兩個模型的權重，下載時間不算推論時間；本地磁碟建議至少保留 30GB 空間，網路速度會影響首次執行時間。

| 步驟 | 要完成的事情 | 指令（專案根目錄執行） |
|---|---|---|
| 1 | 切換 Quiz 1 使用的環境 | `conda activate MMIP_LLM` |
| 2 | 補上 CUDA 12.8 對應的 torchvision；保持 torch 2.9.1 | `python -m pip install torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128` |
| 3 | 安裝圖片處理與 SmolVLM 所需套件 | `python -m pip install -r week6/quiz2/requirements.txt` |
| 4 | 驗證既有 20 題圖像資料 | `python week6/quiz2/Code/prepare_benchmark.py` |
| 5 | 檢查環境、鎖定兩個模型版本，先測圖片處理器；只下載小型設定與 tokenizer，尚未下載權重 | `python week6/quiz2/Code/setup_vlm.py --check-processors` |
| 6 | 下載並執行 Qwen 全部 20 題 | `python week6/quiz2/Code/run_vlm.py --model qwen` |
| 7 | 下載並執行 SmolVLM 全部 20 題 | `python week6/quiz2/Code/run_vlm.py --model smol` |
| 8 | 檢查兩模型是否各 20 題、圖與提示是否一致 | `python week6/quiz2/Code/quiz2_tasks.py status` |
| 9 | 計算 18 題客觀指標 | `python week6/quiz2/Code/quiz2_tasks.py score` |
| 10 | 匯出商用模型的 20 組原圖＋匿名雙答案 | `python week6/quiz2/Code/quiz2_tasks.py export-judge` |
| 11 | 商用模型人工審核：每題新對話，上傳該題 image.png 並貼 prompt.txt；回覆 JSON 存回 responses/ | 見 `week6/quiz2/Output/Commercial_manual/README.md` |
| 12 | 填實際評審模型名稱與日期（不要把範例字串當成名稱）；匯入全部 40 筆評分 | `python week6/quiz2/Code/quiz2_tasks.py import-judge --model-label "填實際模型名稱" --review-date "YYYY-MM-DD"` |
| 13 | 產生比較報告與 20 分要求檢核 | `python week6/quiz2/Code/quiz2_tasks.py report` |

第 12 步請把引號內替換為**介面實際顯示的支援看圖之商用模型名稱**與當天日期；也可先填 `Output/Commercial_manual/session.json` 的 `evaluator_model_label`、`review_date`，然後執行不帶參數的 `import-judge`。

商用模型可使用你能手動上傳圖片的 Gemini／ChatGPT／Claude 版本。必須讓評審**真的看到圖片**；只有貼文字答案無法完成這份視覺評估。每題一次評 A/B，固定同一評審版本。若模型表示看不到圖，重新上傳，不能自行將 `image_seen` 改成 true。整個流程不呼叫 API，也不用 API key。

### 一次跑完本地部分的腳本

先完成安裝套件的步驟 1～3，再於專案根目錄的 **Git Bash** 執行：

```bash
bash week6/quiz2/run_quiz2.sh
```

若在 PowerShell，請明確選擇 Git Bash，避免 Windows 把 `bash` 解析成 WSL：

```powershell
& "C:/Program Files/Git/bin/bash.exe" "week6/quiz2/run_quiz2.sh"
```

腳本依序驗證環境／圖像、鎖版本與檢查處理器、執行兩個模型、計算客觀分數、匯出人工審核包；保留已成功的模型結果，不覆寫。最後會停在「等待商用模型手動審核」，這一步必須由你上傳圖片完成，之後再執行表格的匯入與報告指令。

### 重新執行與問題排除

- 若有未完成的 `Output/full/qwen.json` 或 `smol.json`，檢視錯誤原因後，明確指定 `--overwrite` 重跑該模型。例如 `python week6/quiz2/Code/run_vlm.py --model smol --overwrite`。本版不做半途續跑；重跑全 20 題避免混合設定。
- 如需另一輪獨立結果，兩個模型與所有評分命令都帶相同 `--runs-dir`，審核另指定新的 `--judge-dir`。不要混用舊審核。
- 只檢查套件與 CUDA：`python week6/quiz2/Code/setup_vlm.py --check-only`。缺 Pillow／num2words／torchvision 時，先完成步驟 2～3。
- TLS 憑證錯誤：程式使用系統信任庫 truststore；企業代理仍須正確安裝信任的憑證，不要停用 TLS 驗證。
- 不要升級成 Transformers 5.x；本專案依 4.57.6 的類別與處理器 API 實作。
- 原生圖像處理器會花一些 CPU 時間；20 道 batch=1 推論也不會讓 RTX 5090 一直滿載，不能僅憑 GPU 使用率判斷是否成功。
- 模型鎖定 `Data/models.lock.json` 於第一次 setup 建立。重跑沿用原版本；`--refresh-lock` 只用於有意重做整個實驗時。

## 需要繳交的檔案

| 檔案／資料夾 | 內容 |
|---|---|
| `Code/`、`requirements.txt`、`run_quiz2.sh`、本 README | 可重現的程式與指令 |
| `Data/` | 20 題、18 張圖、參考答案、來源、資料與模型版本鎖定 |
| `Output/full/qwen.json`、`smol.json` | 兩個模型實際推論紀錄與設定 |
| `Output/full/objective_scores.json`、`commercial_scores.json` | 客觀指標與完整商用評分 |
| `Output/full/comparison.csv`、`comparison.json`、`report.md` | 比較表、逐題答案、評語及實驗報告 |
| `Output/full/completion.json` | 三項要求的 20 分證據檢核；不是授課教師保證得分 |
| `Output/Commercial_manual/` | 20 組原圖與審核提示、原始商用回覆、匿名映射與實際評審名稱／日期 |

`models/` 的模型權重不用提交。商用對話截圖可放 `Output/Commercial_manual/screenshots/`，建議保留畫面中的模型名稱、圖片及一組完整審核回覆。

## 離線工具驗證

```bash
python -m unittest discover -s week6/quiz2/Code -p test_quiz2_tools.py -v
```

測試涵蓋 OCR／IoU、無效與重複 JSON、兩模型結果核對、完整匯出匯入報告流程、空白審核拒絕，以及圖片、提示、設定和答案變動後拒絕沿用舊結果。測試不會下載權重，也不代表實際模型推論已完成。

## 官方參考

- [Qwen2.5-VL 模型卡](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [SmolVLM2 模型卡](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)
- [Transformers 4.57.1 Qwen2.5-VL 文件](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2_5_vl)
