# Gemini 手動評分

共20題，每題含四組匿名回答，共80份回答。

每次開一個新對話，將對應prompts/*.txt完整內容交給你實際使用的Gemini；把回傳JSON存入同名responses/*.json。如有Markdown框，移除外層框後保存純JSON。保留原始商用評分，不自行改成希望的分數；有爭議時另存覆核紀錄。不要上傳manifest.json，它含匿名代碼對照。不要只評挑選的成功案例。

| 題目 | 提示檔 | 回覆檔 |
| --- | --- | --- |
| math_01 | [prompts\01_math_01.txt](prompts\01_math_01.txt) | [responses\01_math_01.json](responses\01_math_01.json) |
| math_02 | [prompts\02_math_02.txt](prompts\02_math_02.txt) | [responses\02_math_02.json](responses\02_math_02.json) |
| logic_01 | [prompts\03_logic_01.txt](prompts\03_logic_01.txt) | [responses\03_logic_01.json](responses\03_logic_01.json) |
| logic_02 | [prompts\04_logic_02.txt](prompts\04_logic_02.txt) | [responses\04_logic_02.json](responses\04_logic_02.json) |
| reading_01 | [prompts\05_reading_01.txt](prompts\05_reading_01.txt) | [responses\05_reading_01.json](responses\05_reading_01.json) |
| reading_02 | [prompts\06_reading_02.txt](prompts\06_reading_02.txt) | [responses\06_reading_02.json](responses\06_reading_02.json) |
| classify_01 | [prompts\07_classify_01.txt](prompts\07_classify_01.txt) | [responses\07_classify_01.json](responses\07_classify_01.json) |
| classify_02 | [prompts\08_classify_02.txt](prompts\08_classify_02.txt) | [responses\08_classify_02.json](responses\08_classify_02.json) |
| extract_01 | [prompts\09_extract_01.txt](prompts\09_extract_01.txt) | [responses\09_extract_01.json](responses\09_extract_01.json) |
| extract_02 | [prompts\10_extract_02.txt](prompts\10_extract_02.txt) | [responses\10_extract_02.json](responses\10_extract_02.json) |
| format_01 | [prompts\11_format_01.txt](prompts\11_format_01.txt) | [responses\11_format_01.json](responses\11_format_01.json) |
| format_02 | [prompts\12_format_02.txt](prompts\12_format_02.txt) | [responses\12_format_02.json](responses\12_format_02.json) |
| translate_01 | [prompts\13_translate_01.txt](prompts\13_translate_01.txt) | [responses\13_translate_01.json](responses\13_translate_01.json) |
| translate_02 | [prompts\14_translate_02.txt](prompts\14_translate_02.txt) | [responses\14_translate_02.json](responses\14_translate_02.json) |
| summary_01 | [prompts\15_summary_01.txt](prompts\15_summary_01.txt) | [responses\15_summary_01.json](responses\15_summary_01.json) |
| summary_02 | [prompts\16_summary_02.txt](prompts\16_summary_02.txt) | [responses\16_summary_02.json](responses\16_summary_02.json) |
| code_01 | [prompts\17_code_01.txt](prompts\17_code_01.txt) | [responses\17_code_01.json](responses\17_code_01.json) |
| code_02 | [prompts\18_code_02.txt](prompts\18_code_02.txt) | [responses\18_code_02.json](responses\18_code_02.json) |
| grounding_01 | [prompts\19_grounding_01.txt](prompts\19_grounding_01.txt) | [responses\19_grounding_01.json](responses\19_grounding_01.json) |
| grounding_02 | [prompts\20_grounding_02.txt](prompts\20_grounding_02.txt) | [responses\20_grounding_02.json](responses\20_grounding_02.json) |

全部保存後執行 import-judge；依終端提示輸入實際Gemini模型名稱及評分日期。
