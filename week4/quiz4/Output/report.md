# Quiz 4：影像描述模型訓練與比較

## 方法

Flickr8k 官方 image-level 分割，每張圖五段描述。固定 ImageNet ResNet-50 編碼器，空間特徵快取；LSTM 與具 cross-attention 的 Transformer Decoder 分別從頭訓練。LSTM 以平均影像特徵初始化 hidden/cell；Transformer 使用空間位置編碼。訓練採 teacher forcing、PAD 忽略交叉熵，依 validation token loss 選擇權重。測試只輸入圖片特徵與 BOS，逐字生成到 EOS 或最大長度，不輸入參考句。

## BLEU 與時間比較

| model | decoding | bleu1 | bleu2 | bleu3 | bleu4 | decoder_mean_ms | estimated_pipeline_mean_ms | mean_generated_tokens | training_seconds | decoder_parameters | total_parameters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lstm | greedy | 0.606316 | 0.400873 | 0.262904 | 0.173703 | 17.081699 | 25.090062 | 12.100000 | 100.900670 | 5626324 | 29134356 |
| lstm | beam | 0.639299 | 0.436638 | 0.300960 | 0.209243 | 22.527867 | 30.536230 | 10.866667 | 100.900670 | 5626324 | 29134356 |
| transformer | greedy | 0.622175 | 0.420969 | 0.284814 | 0.191760 | 26.944431 | 34.952794 | 11.766667 | 54.033021 | 6431700 | 29939732 |
| transformer | beam | 0.647175 | 0.446290 | 0.309990 | 0.213345 | 37.130943 | 45.139307 | 11.000000 | 54.033021 | 6431700 | 29939732 |

本次 BLEU-4 最高為 transformer / beam。BLEU 為 corpus multi-reference BLEU-1～4，使用全部測試影像與每張五段參考句，統一 tokenize、無 smoothing、分數介於 0～1。詞彙表只由訓練描述建立。

推論時間是相同測試樣本、batch=1、暖機後重複測量且 CUDA 同步。decoder_mean_ms 使用預先算好的影像特徵；estimated_pipeline_mean_ms 為另行測量的影像讀取／前處理／編碼平均時間加上 decoder 平均時間，屬估計總和。句長會影響速度，另記錄生成 token 數及 tokens/sec。Gemini 網路時間分開保存。

共用影像特徵擷取耗時 130.06 秒，不重複算入兩個 decoder 的訓練時間。

## Gemini 影像符合度評估

本次繳交採手動上傳 Gemini 網頁的 10 張代表案例，每張獨立評估 LSTM / Transformer 的 greedy 描述，共 20/20 筆，已完整匯入。介面顯示模型/模式：3.1 pro（使用者記錄，非 API 後端版本）。

| model | successful | expected | mean_score_0_to_5 | match_rate | partial_rate | mismatch_rate |
| --- | --- | --- | --- | --- | --- | --- |
| lstm | 10 | 10 | 2.800000 | 0.400000 | 0.300000 | 0.300000 |
| transformer | 10 | 10 | 3.000000 | 0.500000 | 0.200000 | 0.300000 |

案例由人工依內容與錯誤類型挑選，A/B 描述位置平衡隨機；未向評估者提供模型名稱或人工參考句。符合率僅限這十張案例，不能外推全體測試集。先前 API 嘗試及未使用的 100 張手動方案不混入此表。詳見 [完整案例評估報告](gemini_cases10/report.md)、[選例說明](gemini_cases10/CASES.md) 與 responses/ 原始回覆。

兩種模型使用同一批圖片，greedy 描述分別評估；評估主體、動作、場景、錯誤細節，輸出符合度與理由。Gemini 是輔助評估者，判斷可能有誤，仍應人工檢查。summary.json 記錄成功數、預期數、模型版本、prompt 與完成狀態。

## 檔案

各模型包含 best.pt、last.pt、history.csv、learning_curves.png；greedy/beam 目錄包含 predictions.json/csv、bleu.json、timing.json、caption_examples.png。本次手動 Gemini 原始回覆、圖片、A/B 對照與評分位於 gemini_cases10/；先前 API 嘗試保存在本機 gemini/。
