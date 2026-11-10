# Quiz 3 偏好與評估資料

完整說明見 DATASET_CARD.md。LLM／VLM 各有 200 組 train、20 組 dev；另外有 100 題文字目標測試、100 題通用題與 50 張視覺測試圖。
視覺測試每張建立原圖、JPEG、低解析度、暗／亮五個條件，共 250 個輸入。
prompt/chosen/rejected 與偏好理由均保留，標籤由規則與繪圖物件清單產生，不是人工 RLHF 評分。
本週自建小型資料隨 GitHub 提交；manifest.json 凍結檔案雜湊。新增此說明不改寫原有凍結資料。
