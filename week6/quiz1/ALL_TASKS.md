# Quiz1：完成基礎與進階全部30分

GitHub 已包含四組各20題的正式實驗、全部商用評分及平均GPU記憶體原始樣本，結果在 Output/full。以下為重現流程；已有輸出不會直接覆寫，重做前應先另存既有實驗。

## 先準備同一個PowerShell視窗

```powershell
conda activate MMIP_LLM
Set-Location ./week6\quiz1
$env:PYTHONUTF8 = "1"
$qwenRevision = (Get-Content Output/full/bf16_basic.json -Raw -Encoding utf8 | ConvertFrom-Json).revision
```

確認 $qwenRevision 有值，並在四組實驗都沿用它。本次低精度使用4-bit NF4；四組皆固定相同生成參數、題目順序與42+題號索引的種子。每組啟動單獨進程，依序執行。

## 執行步驟

| 步驟 | 工作與完成條件 | 指令（PowerShell） |
| --- | --- | --- |
| 1 | 安裝GPU記憶體取樣套件 | `python -m pip install -r requirements_extra.txt` |
| 2 | A：BF16＋基本提示，20題並量測平均／峰值GPU記憶體 | `python Code/qwen_local.py --precision bf16 --revision $qwenRevision --questions Data/benchmark.json --profile --output Output/full/bf16_basic.json` |
| 3 | B：BF16＋優化提示，20題並量測記憶體 | `python Code/qwen_local.py --precision bf16 --revision $qwenRevision --questions Data/benchmark.json --optimized --profile --output Output/full/bf16_optimized.json` |
| 4 | C：4-bit NF4＋基本提示，20題並量測記憶體 | `python Code/qwen_local.py --precision nf4 --revision $qwenRevision --questions Data/benchmark.json --profile --output Output/full/nf4_basic.json` |
| 5 | D：4-bit NF4＋優化提示，20題並量測記憶體，完成進階結合實驗 | `python Code/qwen_local.py --precision nf4 --revision $qwenRevision --questions Data/benchmark.json --optimized --profile --output Output/full/nf4_optimized.json` |
| 6 | 確認四組各20題，profile=True | `python Code/quiz1_tasks.py status` |
| 7 | 計算10道完全客觀題的量化分數 | `python Code/quiz1_tasks.py score` |
| 8 | 匯出20份Gemini手動評分提示，每份含四組匿名回答，共80份回答 | `python Code/quiz1_tasks.py export-judge` |
| 9 | 人工操作：閱讀指南，逐一將prompts內容交給Gemini，將回覆JSON存回同名responses檔 | `notepad Output/Gemini_manual/README.md` |
| 10 | 匯入全部實際商用評分；終端會詢問模型名稱與評分日期 | `python Code/quiz1_tasks.py import-judge` |
| 11 | 產生比較表、分析報告與六項完成檢查 | `python Code/quiz1_tasks.py report` |
| 12 | 確認all_requirements_complete=true，且六個checks全部true | `Get-Content Output/full/completion.json -Raw` |
| 13 | 開啟繳交用分析報告 | `notepad Output/full/report.md` |

四組總共80筆回答。步驟9是實際操作Gemini，開啟README不等於已完成評分；全部20份response都填妥才能匯入。沒有API呼叫或自動代填商用評分。

每份prompt已附原題目、參考答案與評分規則；請開新對話並貼入完整內容，不上傳含代碼對照的manifest.json。Gemini若回覆Markdown程式碼框，存入responses時只保留純JSON。請記錄介面顯示的實際評分模型名稱。

步驟11在商用評分或記憶體量測不足時會產生未完成報告並回傳exit 3。依completion.json補齊，不要把null分數改成0來冒充完成。

## 對應投影片

| 作業要求 | 對應成果 |
| --- | --- |
| 5分：部署開源LLM | demo.json及四組Qwen本機推論，記錄模型revision、GPU與套件 |
| 5分：20題Instruction Benchmark | Data/benchmark.json、題目、參考答案、評分規則，四組都評全部20題 |
| 5分：提示工程與評估 | A對B，結構化提醒與部分Few-shot，客觀題與商用分數 |
| 5分：量化，使用基本Prompt並以量化指標與商用LLM評估 | A對C，BF16與4-bit，同題目與基本提示 |
| 5分：預設精度平均GPU記憶體 | A的bf16_basic.memory.json，原始取樣與時間加權平均 |
| 5分：低精度＋優化Prompt、記憶體與成效比較 | D對A，另可D對C判斷低精度下的提示效果 |

課程未提供完整提示技巧投影片，請核對結構化提示與Few-shot是否對應課堂介紹的方法。若要更換提示，需另存題庫版本並重跑四組，不能在已完成測試後僅替某一組改題。

## 量測與評分範圍

- 每次載入後固定暖機一次；只取樣prefill與回答生成期間，排除下載、模型載入、tokenization與題目間I/O。預設每50ms取樣，保留實際時間戳記，以時間加權計算平均值。
- NVML記錄整張GPU使用量，包含桌面及其他程式；請讓四組執行時的其他GPU工作一致。PyTorch allocated/reserved與峰值另列，不能混為同一指標。載入前整卡基準也會保存，差值只作估計。
- 10道完全客觀題可自動計分；全部20題由Gemini依規則另行評分。程式題在商用評分中作靜態審閱，未在主機執行模型產生的程式，因此不宣稱執行測試通過率。
- 比較表保留空回答、輸出token上限案例、輸入／輸出token數與耗時；低精度不一定較快，優化提示不一定提高分數。若要調整256的輸出上限，四組應一致重新執行並重新匯出商用評分資料。

## 重跑與保留結果

更新後的qwen_local.py若遇到已存在的輸出會停止，避免覆寫。需要重跑整組時，可改用新實驗目錄，例如 Output/full_v2，但四個檔名仍須保持一致；後續工具全部加 --runs-dir Output/full_v2，並搭配新的 --judge-dir Output/Gemini_manual_v2。不要同時啟動多組模型。

原版程式已保留為 Code/qwen_local.before_profile.txt。先前未取樣的回答仍留在Output根目錄，可供個人參考；本流程的正式四組使用新的Output/full。

## 繳交成果

- Code/qwen_local.py、gpu_memory.py、quiz1_tasks.py與環境資訊。
- Data/benchmark.json、兩版提示、參考答案、評分規則及manifest。
- Output/full/四組回答JSON與四個*.memory.json原始量測。
- objective_scores.json、gemini_scores.json、comparison.csv/json、report.md、completion.json。
- Output/Gemini_manual/中的實際商用評分prompt與response，以及評分模型名稱／日期。

## 程式檢查紀錄

本次新增工具的6項離線測試通過，涵蓋時間加權取樣、客觀題判分、手動評分匯入、過期結果拒絕、記憶體資料核對、完成狀態與模擬推論主流程。測試使用臨時合成資料及模擬模型／GPU，沒有寫入正式成果，沒有替你執行Qwen或呼叫Gemini。正式硬體量測與商用評分仍須依上表完成。

```powershell
python -m unittest discover -s Code -p test_quiz1_tools.py -v
```
