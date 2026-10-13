# Week 4 執行結果審查

結論：六個本機模型的結果可重算、整體合理；需要說明 IMDb 少量來源重複、
情感 LSTM 的過擬合與過度自信，以及 caption 品質／速度的取捨。
Gemini 後續改採使用者指定的十張案例展示：兩模型共 20/20 筆網頁手動評分已匯入。
原 API 方案仍停在 10/200 筆，兩種方式分開保存；現在 completion.json 已標記所選作業範圍完成。

前次審查只做已訓練權重的推論與結果檢查，沒有重訓或改變資料分割。
此文件已補上後續 Gemini 手動案例評估狀態。
重算資料見 results_audit.json、followup_checks.json，程式見同目錄兩份 Python。

## 核對範圍

- 四個分類模型的儲存機率重新計算 Top-1、Top-5、Macro-F1、Macro-AUC，與原 metrics.json 一致。
- 重新載入四個 best.pt，對完整 validation/test 再推論，test 的分類結果皆沒有改變。
  ResNet 的個別機率最大差約 0.0038，符合本流程非逐位元確定性的數值差異，17,778 個 argmax 都相同。
- 檢查 best.pt epoch 與 complete.json / history.csv 一致，測試實際使用選出的最佳權重。
- 獨立實作重算四組 corpus BLEU，差異只在浮點尾數（小於 1e-12）。
- 四組 caption 的 1,000 個 test ID 與五段參考描述全部對應正確，沒有空描述或 <unk>。
- Caption 抽樣 20 張重新推論：LSTM greedy/beam 與 Transformer greedy 均 20/20 相同；
  Transformer beam 在 batch=1 為 19/20，改回原本 batch=128 對全部 1,000 張重跑則 1000/1000 相同。
  差異與 batch / FP16 beam 排序數值敏感性相容，不是對錯圖片或讀錯權重的證據。
- 檢查資料 ID、IMDb 完整原始文字 hash、OrganAMNIST 全部影像像素 hash、
  Flickr8k 全部 JPEG 檔案 hash 的跨集合重疊。
- 人工檢視 caption_examples.png 與實際 Gemini 回應。

## Quiz 1：資料分割

| 資料集 | Train | Validation | Test | 核對結果 |
| --- | --- | --- | --- | --- |
| IMDb | 20000 | 5000 | 25000 | ID 不重疊，但來源含少量完全相同評論 |
| OrganAMNIST | 34561 | 6491 | 17778 | 跨集合沒有完全相同的 28×28 像素 |
| Flickr8k | 6000 | 1000 | 1000 | ID 和 JPEG 位元組跨集合均不重複，每圖五段參考句 |

IMDb 原始內容重複情況：

- train / validation：26 個共用文字 hash，涉及 28 筆 validation。
- train / test：103 筆 test 與 training 完全相同，占 test 0.412%。
- validation / test：20 筆 test 與 validation 完全相同。
- 合併排除 training 或 validation 重複內容後，test 剩下 24,877 筆。

原流程遵循官方 test 與固定 train/validation 分割，但沒有額外對 IMDb 做內容去重。
因此報告不宜宣稱三組資料「內容完全獨立」。
以下是同一份已訓練模型在排除重複測試樣本後的敏感度分析，沒有重新訓練：

| 模型 | 官方 25000 筆 Accuracy | 排除 123 筆後 Accuracy | 排除後 Macro-F1 |
| --- | --- | --- | --- |
| RNN | 75.6160% | 75.6442% | 75.5805% |
| LSTM | 83.3760% | 83.3179% | 83.2901% |

LSTM 約降低 0.058 個百分點，主要結論不變。
這個補充分析不能消除 training/validation 的來源重複，也不能證明沒有近似內容。
若未來重做嚴格獨立實驗，可先依文字 hash 分組再切分。
影像檢查涵蓋「完全相同」內容，不代表已排除所有近似影像；NPZ 沒有逐筆病患 ID，
本次無法另外從原始影像檔自行重建病患分組。

## Quiz 2：RNN 與 LSTM

| 模型 | 選用 epoch | Train Accuracy（該 epoch） | Validation Accuracy | Test Accuracy | Test 正類 AUC |
| --- | --- | --- | --- | --- | --- |
| RNN | 5 | 74.615% | 75.040% | 75.616% | 0.816201 |
| LSTM | 14 | 99.460% | 84.060% | 83.376% | 0.901322 |

RNN 的選用 epoch 在 train/val/test 上一致，之後 validation 表現震盪／下降，
early stopping 與載入 epoch 5 最佳權重的行為正確。

LSTM 比 RNN 高 7.76 個百分點；test 與 validation 只差約 0.68 個百分點，
分類優勢可重現於此次測試集。但 train 與 validation 差 15.40 個百分點，有明顯過擬合。
validation loss 在 epoch 6 最低約 0.439，之後升至選用 epoch 14 的 0.946，
Accuracy 仍升至 84.06%。這是「以 validation Accuracy 選權重」允許出現的結果，
不是挑錯 checkpoint。

額外用儲存機率檢查：LSTM test NLL 約 0.995，高於 RNN 約 0.579；
LSTM 錯誤預測的平均最高機率約 93.61%，顯示錯誤時仍相當自信。
所以應寫「LSTM 分類較好，但過擬合且機率偏過度自信」，不能只展示 train 接近 100%。
若日後改善，可另以 validation loss 選權重、增加正則化或做信心校準對照；
這些並不保證 Accuracy 一定提高，本次沒有改動或重訓。

二分類的 Top-5 必然為 100%，本題比較 Accuracy / F1 / 正類 ROC-AUC 才有意義。

## Quiz 3：ViT 與 ResNet

| 模型 | Validation Top-1 | Test Top-1 | Test Top-5 | Test Macro-AUC | Test 錯誤數 |
| --- | --- | --- | --- | --- | --- |
| ViT-B/16 | 99.7535% | 96.6419% | 99.9213% | 0.998868 | 597 |
| ResNet-50 | 98.9986% | 95.7982% | 99.9213% | 0.998569 | 747 |

ViT 此次較高約 0.844 個百分點，少錯 150 張；Macro-AUC 差僅約 0.000299。
高 AUC 與較低 Top-1 沒有矛盾：AUC 衡量每類相對其他類的機率排序，
Top-1 衡量最後類別是否選對。one-vs-rest 中許多負類容易區分，AUC 可以很高，
仍有部分器官在 argmax 上混淆。

ViT 的 kidney-left recall 約 88.23%、F1 約 89.95%，是值得討論的弱項。
主要混淆：kidney-left → spleen 123 張、kidney-right → kidney-left 83 張、
kidney-left → pancreas 60 張。肺／肝等類別表現較高，整體平均會掩蓋這個落差。

ViT train 接近 100%、validation 99.75%，test 96.64%；ResNet 也有 train/test 落差。
這表示需保留泛化限制的討論，但完全相同像素檢查未發現跨集合洩漏。
只有此次 seed 與分割，不能宣稱 ViT 在所有情況都勝過 ResNet。

參數：ViT 約 85.81M，ResNet 約 23.53M；整個 test 推論約 5.94 秒與 3.02 秒。
此次取捨為 ViT Accuracy 略高、ResNet 較小且推論較快。
批次測試秒數包含資料載入／傳輸，不等於單張服務延遲。

## Quiz 4：Captioning

BLEU 以下以 0～100 表示（原 JSON 使用 0～1），並非「描述正確率」。

| 方法 | BLEU-4 ×100 | Decoder 平均延遲 | 估計含編碼平均延遲 |
| --- | --- | --- | --- |
| LSTM / greedy | 17.3703 | 17.08 ms | 25.09 ms |
| LSTM / beam | 20.9243 | 22.53 ms | 30.54 ms |
| Transformer / greedy | 19.1760 | 26.94 ms | 34.95 ms |
| Transformer / beam | 21.3345 | 37.13 ms | 45.14 ms |

兩個模型加 beam 後分數提高、速度變慢，符合本次實際測量。
Transformer / beam 比 LSTM / beam 只多約 0.4102 BLEU 點，decoder 平均延遲約 1.65 倍。
若重視此硬體與實作下的速度／品質平衡，LSTM / beam 有合理取捨；
不能由單次 0.41 點差距推論架構普遍或顯著較好。

人工抽看可正確描述泳池裡的狗、雪地裡的狗；仍會把跳舞說成坐著、
把一隻狗說成兩隻、對衣著／動作加入不存在的細節。BLEU 與這種部分正確的品質相符，
不能將 BLEU-4 21.33 解讀成 21.33% 圖片完全描述正確。

LSTM / beam 的 1000 張有 853 種不同描述，Transformer / beam 有 862 種，
沒有全部輸出同一句的現象，但常見場景仍會套用通用句子。
所有模型皆使用相同影像特徵識別，評估參考句與 test ID 一致。

最佳 epoch：LSTM 26/30，Transformer 9/14。Transformer 從 epoch 10 後
validation loss 不再改善，early stopping 正確在 14 停止並使用 9。
LSTM 後期也有輕微 train/validation gap。
console 中的 caption Accuracy 是 teacher-forcing token accuracy，不是整句描述正確率。

Transformer 訓練共約 54 秒、LSTM 約 101 秒，主要因分別跑了 14 和 30 epochs。
每 epoch 平均約 3.86 秒與 3.36 秒，不能因此說 Transformer 訓練效率是 LSTM 的兩倍。
訓練的並行 token 計算與推論逐字生成不同，訓練總時間較短但逐張描述延遲較長並不矛盾。
共用特徵擷取另約 130 秒，兩者皆未重複加入 decoder 訓練秒數。

延遲測量限相同前 30 張 test、重複 3 次、batch=1、暖機後 CUDA 同步。
estimated_pipeline_mean_ms 是分開量測的 encoder/decoder 平均值總和，
不是實際端到端每張樣本逐筆計時；不宜拿它主張精確端到端 SLA。
目前生成實作沒有增量解碼快取，速度比較適用於本次程式與硬體。

## Gemini：十張案例展示已完成

本次繳交使用十張人工挑選案例、兩模型共二十段 greedy 描述，兩批原始 Gemini 網頁回覆皆通過匯入檢查。
使用者記錄介面模型/模式為 3.1 pro，操作人 JerryWu，審核開始日期 2026-10-13；
這是介面紀錄，沒有宣稱 API 回傳相同後端版本。

| 模型 | 成功／預期 | 平均分數 | match | partial | mismatch |
| --- | --- | --- | --- | --- | --- |
| LSTM | 10/10 | 2.8/5 | 4/10 | 3/10 | 3/10 |
| Transformer | 10/10 | 3.0/5 | 5/10 | 2/10 | 3/10 |

因為刻意涵蓋不同錯誤類型，符合率只描述這十張案例，不能推廣至全部測試集。
每圖 A/B 獨立評分，兩模型可以同時符合或不符合；沒有混入先前 API 的十筆成功結果。
實際原圖、提示詞、原始回覆、對照與分數見 ../quiz4/Output/gemini_cases10/。
completion.json 已依使用者指定的案例展示範圍標記完成。

### 先前 API 嘗試（歷史紀錄）

原 API 方案預期兩個模型各 100 張，共 200 筆；該方案保留的結果為：

- LSTM：10 筆，4 match、4 partial、2 mismatch，平均 3/5。
- Transformer：0 筆。
- 第 11 筆持續 HTTP 429，有限重試後停止，本機模型結果已保存。

早期 errors.json 只保存狀態碼；改善診斷後，後續回應明確指出每專案每模型每日免費請求上限為 20。
API 流程因此停止，後續改用網頁手動案例評估完成作業，並保留原始 API 結果。
官方說明：https://ai.google.dev/gemini-api/docs/troubleshooting

這 10 筆是原本 100 張抽樣清單排序後的前段，不是另外抽出的完整代表樣本。
不能將 40% match rate 推廣到全部 test，也不能與尚未評估的 Transformer 比較。
這項原 API 方案沒有標記完成；整份作業使用上述手動案例結果完成 Gemini 要求。

在可用配額恢復、原本 GEMINI_API_KEY 已設定的 Git Bash 中，只補評估：

    bash week4_homework/train_rtx5090.sh --stage gemini

若 terminal 已位於 week4_homework，使用 bash train_rtx5090.sh --stage gemini。
成功的 10 筆會依相同 keyless cache identity 重用，不需要重新訓練六個模型。

## 整體時間與交付判斷

本次 log 記錄整個流程約 2215.5 秒，即 36 分 55 秒。
各模型 training_seconds 主要是 epoch 的訓練與驗證迴圈時間，不含所有下載、
解壓、checkpoint 存檔、作圖、推論、Gemini 網路等待，因此不能直接相加作為總耗時。

目前本機成果可以使用，沒有發現必須全部重訓的評估錯誤。
Gemini 十張案例已補齊；繳交報告應保留 IMDb 重複的敏感度分析、LSTM 過擬合、
ViT 器官混淆及 caption 品質／速度限制。

BLEU 方法來源：https://aclanthology.org/P02-1040/
