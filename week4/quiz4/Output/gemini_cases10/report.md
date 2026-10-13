# Gemini 手動上傳評估

完整評估已匯入。 成功 20/20 筆。

| model | successful | expected | mean_score_0_to_5 | match_rate | partial_rate | mismatch_rate |
| --- | --- | --- | --- | --- | --- | --- |
| lstm | 10 | 10 | 2.800000 | 0.400000 | 0.300000 | 0.300000 |
| transformer | 10 | 10 | 3.000000 | 0.500000 | 0.200000 | 0.300000 |

操作人：JerryWu；開始日期：2026-10-13；Gemini 介面模型/模式：3.1 pro。此名稱由使用者記錄，非 API 回傳的後端版本。

人工挑選十張不同內容與錯誤類型的展示案例，非隨機抽樣。每張獨立評估匿名 A/B 描述，A/B 位置平衡隨機，未提供人工參考句。每次五張、每批新對話。若操作方式不同，應於 session.json notes 註明。網頁評估與 API 評估的條件不同，未混算先前 API 成功結果；本批符合率不代表全體測試集正確率。

## 缺漏或錯誤

無。

原始回覆位於 responses/，人工保存的截圖/PDF位於 evidence/，A/B 對照位於 manifest.json。
