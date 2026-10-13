# 手動上傳 Gemini 審核

共 10 張固定測試圖片、2 批，每張兩段 greedy 描述，共 20 筆評估。
這是 Gemini 網頁的人工操作評估，與 API 評估分開報告。請完整評估這批資料，不跳過難例。
本批是人工挑選的 10 張展示案例，涵蓋不同內容與錯誤類型。選例理由見 CASES.md；這份說明不要提供給 Gemini，避免引導評分。符合率僅限這十張案例，不代表隨機樣本或全體測試集。


1. 在 Gemini 網頁選擇一個可看圖的模型/模式，記下介面顯示名稱。所有批次使用同一模式。
2. 開啟 uploads/batch_01，選取其中五個 JPG 檔上傳；不要上傳整個 ZIP、manifest.json 或其他批次。
3. 複製同資料夾 PROMPT.txt 全文到訊息框，確認五張圖都已附上後送出。
4. 複製 Gemini 的完整 JSON 回覆，以 UTF-8 純文字存成 responses/batch_01.txt。
   也可用同名 .json，但同一批只保留一份；不要用 Word，也不要自行補寫 Gemini 的答案。
5. 每批開新對話，依序完成至 batch_02；對應存成同名文字檔。
   可把含圖片、提問與回覆的截圖/PDF留在 evidence/，檔名標記批次。
6. 人工抽查圖片對應、分數理由與幻覺。若 Gemini 看錯檔名或漏評，補正該批後保存完整回覆；
   若你另有不同意見，另存 evidence/ 的備註，不要偷偷改寫 Gemini 原始評分。
7. 編輯 session.json：填 evaluator_model_label（介面實際顯示名稱，不猜後端版本）、
   operator（你的名字）、review_started_on（開始日期 YYYY-MM-DD）。不要中途換評估模式。
8. 在 D:\projects\MMIP 的 MMIP 終端機執行：

   python week4_homework/quiz4/Code/manual_gemini.py --case-study --import-all

只做完幾批也可匯入看進度；漏評或格式錯誤會列出批次與原因，未滿 20 筆不會算完成。
全部通過後會更新本資料夾 summary.json、report.md、Quiz4 報告與整份作業完成狀態。
資料夾：week4_homework/quiz4/Output/gemini_cases10/。
舊 API 的 10 筆保留但不混入這次統計。只需重做 Gemini 評估，不需重訓任何模型。

## 如何對應圖片

使用原始 JPG 檔名對應 PROMPT 的 image_id，A/B 是兩模型描述的平衡隨機順序。
不要把模型名稱或人工參考句提供給 Gemini，以免影響判斷。
如果介面無法辨識檔名，先請它核對附圖與候選描述的對應，確認無誤後再評分。
若單批多圖仍無法可靠對應，請把同批五張分別發送審核，最後合併十筆原始判斷為同一批 JSON，
保留 batch_id；只整理結構，不修改評分內容，並在 session.json notes 註明。

Gemini 網頁仍有自己的使用限制。達到上限可稍後續做；本工具不呼叫 API，不需要 API Key。
若更換模式或模型，需另開一份評估，避免將不同評估者的結果混算。

## 繳交

包含 report.md、summary.json、manifest.json（本機 A/B 對照）、session.json、responses/ 原始回覆、
imported/ 驗證結果，以及 evidence/ 證據。說明評估是「手動上傳圖片至 Gemini」，
不能寫成自動 API 已完成，也不要把這批的符合率當成全體測試集正確率。
