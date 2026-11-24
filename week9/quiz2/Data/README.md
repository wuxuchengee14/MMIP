# 訓練與基準資料

`benchmark.jsonl` 保存 38 筆單輪、10 筆多輪與 8 筆圖片題的輸入及參考答案。圖片題使用已凍結的 OCR 文字。`train_rows.json` 供訓練程式切分，`sft.jsonl` 和 `dpo.jsonl` 是原課堂資料的對應格式。本次正式實驗 B 使用 LoRA SFT。[方法與結果](../Output/report.md)
