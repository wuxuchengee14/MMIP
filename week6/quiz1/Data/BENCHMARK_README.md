# Quiz1 Instruction Benchmark：使用方式

本題庫共20題、10種能力各2題，適用於現有 Code/qwen_local.py。資料由本次作業自訂，不需外部知識、連網搜尋或額外資料集。

## 檔案

- benchmark.json：實際程式輸入，是20筆物件的JSON陣列。
- benchmark_questions.md：完整基本題目，便於閱讀。
- benchmark_optimized.json：獨立的20題優化版本，執行時必須加 --optimized。
- benchmark_optimized_prompts.md：完整優化提示，便於逐題閱讀。
- benchmark_optimized_manifest.json：優化版本與來源題庫的SHA256對應。
- benchmark_answers.md：參考答案與逐題評分規則。
- benchmark_manifest.json：題庫版本、SHA256與結構檢查結果。

每筆包含 id、category、difficulty、prompt、optimized_prompt、prompting_techniques、reference、grading。
現有推論程式只會將選取的 prompt 或 optimized_prompt 傳給模型；reference、grading 只保留供事後評估。

## 獨立 Optimized Prompt 版本

benchmark_optimized.json 與原題庫有相同的20個題號、參考答案及評分規則。優化提示沿用原題庫已有的 optimized_prompt，這次只另外匯出，沒有修改題目或依測試結果調整提示。

獨立檔將基本提示存為 basic_prompt 供對照，實際輸入欄位是 optimized_prompt，因此必須加 --optimized。現有程式不會將 basic_prompt、reference、grading 傳給模型。

在 Quiz1 目錄、已啟用 MMIP_LLM 的 PowerShell 執行：

```powershell
$qwenBenchmarkRevision = (Get-Content Output/demo.json -Raw -Encoding utf8 | ConvertFrom-Json).revision
python Code/qwen_local.py --precision bf16 --revision $qwenBenchmarkRevision --questions Data/benchmark_optimized.json --optimized --output Output/bf16_optimized_standalone.json
```

也可以繼續使用原本的 Data/benchmark.json 加 --optimized，兩者選取的20段模型提示完全相同。不要把這兩次執行當成兩種不同的提示技巧實驗。

## 覆蓋範圍

| 題號 | 能力 | 內容 |
| --- | --- | --- |
| math_01–02 | 數學計算 | 折扣運費、集合重複計數 |
| logic_01–02 | 邏輯推理 | 排序限制、必然與未知 |
| reading_01–02 | 閱讀理解 | 因果限制、跨規則日期判斷 |
| classify_01–02 | 文字分類 | 轉折情緒、客服需求 |
| extract_01–02 | 資訊抽取 | 日期與缺漏值、訂單與總額 |
| format_01–02 | 指令與格式遵循 | 行數字數與禁用字元、JSON限制 |
| translate_01–02 | 翻譯 | unless條件、未證實語氣 |
| summary_01–02 | 摘要 | 試辦範圍、決議與未通過提案 |
| code_01–02 | 程式能力 | 保序去重、修正邊界與比較條件 |
| grounding_01–02 | 資料依據與不確定性 | 資料不足、錯誤前提與記憶體概念 |

## 基礎三組實驗（PowerShell）

完整基礎＋進階30分的量測與評分流程，請改依 [ALL_TASKS.md](../ALL_TASKS.md) 執行；下方原始三組指令保留作為未取樣的基本操作說明。新程式遇到既有輸出會停止。

先啟用已建立的環境，切換到 Quiz1，沿用成功部署時保存的模型 revision：

```powershell
conda activate MMIP_LLM
Set-Location D:\projects\MMIP\week6_homework\quiz1
$env:PYTHONUTF8 = "1"
$qwenBenchmarkRevision = (Get-Content Output/demo.json -Raw -Encoding utf8 | ConvertFrom-Json).revision
```

每組各20題。先確認下列輸出檔尚無需要保留的舊結果；有舊結果時改用新的 --output 檔名。

```powershell
python Code/qwen_local.py --precision bf16 --revision $qwenBenchmarkRevision --questions Data/benchmark.json --output Output/bf16_basic.json
python Code/qwen_local.py --precision bf16 --revision $qwenBenchmarkRevision --questions Data/benchmark.json --optimized --output Output/bf16_optimized.json
python Code/qwen_local.py --precision nf4 --revision $qwenBenchmarkRevision --questions Data/benchmark.json --output Output/nf4_basic.json
```

同一時間只跑一組。題目及參考程式碼的字數配合目前256個新token上限設計；如出現截斷，需記錄並對所有組別一致調整上限，不能只替某組補跑較長答案。

## 評分

- 每題0–1分，共20分；百分制為各題得分平均乘100。
- 每類分數為該類兩題平均。分類、抽取與程式題有多個子項，但每整題仍最高1分，避免某類因子項較多而占更高權重。
- exact_match：只移除首尾空白再比對。
- json_fields：依各題的合法JSON、欄位與型別規則評分，不以JSON鍵的排列順序判錯。
- rubric：按列出的語意與格式權重評分，接受語意等價表達。
- python_tests：依附上的案例與輸入不得被修改的規則評分；執行模型產生的程式時應使用受限制的測試環境。
- 翻譯、摘要及開放式回答可交給商用LLM依同一規則評分，保留模型名稱、日期與原始結果。量化比較須同時有量化指標與商用LLM評分。

目前 qwen_local.py 負責生成及保存回答，尚未自動實作上述評分；題庫的 grading 是評分規格，不是已完成的評分結果。

## 提示工程與比較限制

基本與優化提示包含相同正式題目、資料及輸出限制；優化提示另加入結構化提醒與部分不同題目的few-shot示例。這些技巧須再對照課程實際介紹的名稱。

先用額外開發題調整提示，再凍結正式題庫，避免依這20題的測試結果逐題改寫提示。修改題庫後應另存版本並重新生成全部比較組，不能混用不同版本的題目。

這20題是小型自訂評估；結果代表這組題目的表現，不足以推論模型在所有任務上的能力。模型產生空答案、格式錯誤或截斷時均應保留該題，不從分母中刪除。

