# Quiz 3 自建資料集

文字優化目標：來源忠實的 JSON 欄位抽取，缺漏使用 null，資料中指令不得覆蓋任務。
視覺優化目標：只計指定顏色＋形狀；不存在的物件回答 0，避免虛構目標。

兩種模型各有 200 組 train、20 組 dev 的 prompt/chosen/rejected 配對；label_source 與 preference_reason 記錄偏好依據。
這些是自建規則標註偏好，不是人工評分或外部商用模型產生的標籤。chosen/rejected 可用於偏好對齊；本實驗使用 DPO，不宣稱執行 PPO。

LLM 測試另外 100 題，通用 Benchmark 再另外 100 題，十種能力各十題。後者不包含本次優化的 JSON 擷取格式。
VLM 測試另外 50 張圖（目標計數 0/1/2/3/4 各十張），每張衍生五個條件，共 250 個測試輸入。
圖像有精確 objects 清單與 expected_count，所有形狀不重疊、不出界；避免用自然照片描述推斷『未提及就是不存在』的錯誤負例。

只有 train 用於參數更新，dev 用於挑選最低偏好 loss checkpoint。test 與通用題禁止用於訓練、挑選 checkpoint 或追求較高分反覆調參。
manifest.json 凍結所有檔案雜湊；程序檢查提示重複、圖像重複、訓練測試交集、候選是否相同。
變形條件 original / jpeg_q15 / lowres_64 / dark_050 / bright_150 只出現在測試。

所有文本、控制圖與參考答案均由本專案建立，不需要下載新資料或另付 API 費。程式自建的數值／繪圖標籤仍須與實際生成圖片一併閱讀。
這是小型診斷型 Benchmark，不是標準資料集，結果不能推論自然照片或所有通用任務的表現。
