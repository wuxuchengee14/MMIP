# Quiz 2 視覺題庫

20 題、10 種能力、18 張不同圖片：4 張沿用 Week 4 Flickr8k test split 的照片與 14 張原創控制圖。
題目與參考見 benchmark_questions.md；圖片總覽見 contact_sheet.png；來源與原圖 SHA-256 在 sources.json。
不同題型有重用自然照片，不能把二十題當作二十張不同圖片。原始照片權利仍屬原攝影者。
固定的小型測試圖片一併提交；不提交完整 Flickr8k、模型下載或權重。
已有 benchmark.json 時 prepare_benchmark.py 只驗證資料，無須再取得 Week 4 原始照片；重新建構才需要指定來源圖片路徑。
