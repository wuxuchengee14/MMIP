# Week 7 配分對照

| Quiz | 要求 | 分數 | 證據 |
|---|---|---:|---|
| 1 | 開源 LLM 部署 | 5 | quiz1/Output/full 四組 Qwen3 結果 |
| 1 | 二十題 Instruction Benchmark | 5 | quiz1/Data/benchmark.json |
| 1 | 提示工程與評估 | 5 | 基本／優化提示、objective_scores.json、gemini_scores.json |
| 1 | 低精度量化與客觀／商用評估 | 5 | BF16 與 NF4 的相同題目及原始評分 |
| 1 | 預設精度平均 GPU 記憶體 | 5 | bf16_basic.memory.json 原始樣本與積分 |
| 1 | 量化＋優化提示的品質／記憶體比較 | 5 | nf4_optimized 與 comparison.csv |
| 2 | 部署執行兩個開源 VLM | 10 | quiz2/Output/full/qwen.json、smol.json |
| 2 | 二十道視覺 Benchmark | 5 | quiz2/Data/benchmark.json、images/ |
| 2 | 同提示、量化指標與商用看圖審核 | 5 | quiz2/Output/full 與 Commercial_manual/ |
| 3 | LLM 優化方向與至少一百偏好資料 | 5 | quiz3/Data/llm_train.json（200） |
| 3 | 另外一百無關通用題 | 5 | quiz3/Data/general_100.json |
| 3 | 實際微調 LLM 與兩種能力前後量化 | 10 | quiz3/Output/llm/、metrics.json |
| 3 | VLM 優化方向與至少一百偏好資料 | 10 | quiz3/Data/vlm_train.json（200） |
| 3 | 實際微調 VLM 與前後量化 | 5 | quiz3/Output/vlm/、metrics.json |
| 3 | 畫質與明暗影響 | 5 | quiz3/Data/variants/、五條件表與圖 |

合計 90 分的實作證據涵蓋；實際分數由教師判定。數據不好看不代表沒執行，也不能當作成功改善；負結果與限制均保留。
