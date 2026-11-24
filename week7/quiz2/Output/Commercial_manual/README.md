# 商用視覺模型手動評估

1. 使用支援上傳圖片的商用模型；記錄介面實際顯示的模型名稱。
2. 每一題開新對話，**同時上傳 cases/vision_XX/image.png 與貼上同目錄 prompt.txt 全文**。
3. 將模型回傳的完整 JSON 存入 responses/vision_XX.json，取代空白模板；共 20 題、40 筆評分。
4. image_seen 必須為 true，且 image_observations 與每筆 reason 都必須填寫。看不到圖時重新上傳，不能自行改為 true。
5. 不要上傳 session.json（含匿名映射）、其他題答案或這份說明。每題同一則回覆應同時評 A/B。
6. session.json 的 evaluator_model_label 與 review_date 填實際模型名稱及 YYYY-MM-DD；也可在 import-judge 時指定。
7. 所有題目用同一個商用模型版本。若更換版本，另建審核目錄重新評估全套。

這是人工操作商用模型，不會呼叫 API，也不需要 API key。模板中的 null 與空白字串不是已完成的評分。
