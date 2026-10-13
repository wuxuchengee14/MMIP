# 驗證紀錄

驗證環境：Windows 11、Python 3.12.14、PyTorch 2.9.1+cu128、
torchvision 0.24.1+cu128、RTX 5090 31.84 GiB。

## 已執行

- 六模型正式訓練、完整測試指標及 BLEU 已產生，獨立重算與推論審查見 analysis/results_review.md。
- 最新離線測試共 25 項通過，包含 HTTP 429/503 退避、成功快取、人工回覆完整性、
  A/B 對照、評估者更換拒絕混算，以及十張案例方式的完成狀態。
- 十張人工挑選案例的 Gemini 網頁原始回覆共 20 筆已通過匯入檢查；API 的 10/200 筆另行保留。
  完整性檢查不代表人工或 Gemini 的每項影像判斷一定正確。

- Python 編譯檢查、四份 Notebook schema 驗證，Notebook 預設不啟動訓練。
- Git Bash 語法與 dry-run、PowerShell dry-run；兩者正確選到 MMIP Conda Python。
- CUDA 卷積、AMP、backward 與 AdamW 步驟。
- 六種實際架構，以設定的 batch size 執行一次 AMP 前向／反向／AdamW；
  使用隨機輸入與未預訓練權重，沒有將這個檢查視為收斂或成績驗證。
- Windows spawn DataLoader、pin memory、uint8 28×28 傳輸與 CUDA 批次前處理。
- IMDb / Flickr8k 兩個壓縮檔 / ViT 權重下載 URL 的 HTTPS HEAD 皆回傳 200。
  Conda 原本有 SSL 驗證錯誤；加入 truststore 使用系統憑證後通過，未關閉憑證驗證。

## 離線測試

執行：

    python -m unittest discover -s week4/tests -p test_workflow.py

涵蓋：

1. 已知答案的多參考 corpus BLEU、重複字截斷、brevity penalty 與空描述。
2. 訓練詞彙表的未知詞、caption BOS/EOS 與安全解壓路徑。
3. RNN / LSTM 有效句尾索引不受尾端 padding 影響。
4. 兩個 caption decoder 的因果性、梯度與 greedy/beam 生成。
5. 中斷續訓後完整最後 epoch 權重與不中斷執行逐位元一致（CPU），拒絕變更設定。
6. 合成資料小型流程：文字兩模型、影像兩模型、caption 兩模型、報表與完成清單。
   此測試使用縮小的影像分類 fixture；完整 ViT/ResNet 另外由 GPU probe 覆蓋。
7. Gemini mock 評估、成功快取重用、缺少 key 時不能宣稱全部完成。
8. 已知完美三分類機率的 Top-1、Macro-F1、Macro-AUC 均為 1。

Gemini mock 只測程式流程，不是真實 Gemini 評估結果。

## GPU 顯存檢查

實測詳見 tests/gpu_probe_results.json。最終設定：

| 模型 | Batch | 峰值 allocated VRAM |
| --- | --- | --- |
| RNN | 128 | 約 0.35 GiB |
| LSTM | 128 | 約 0.28 GiB |
| ViT-B/16 | 128 | 約 9.25 GiB |
| ResNet-50 | 128 | 約 5.97 GiB |
| Caption LSTM | 128 | 約 0.56 GiB |
| Caption Transformer | 128 | 約 0.69 GiB |

這是 PyTorch 分配量，不包含桌面、其他程序與所有驅動保留量。
單步時間含初始化／暖機，不適合直接推算整個作業訓練時間。

## 尚待使用者完整執行

沒有啟動完整三資料集下載、正式六模型訓練或真實 Gemini 呼叫。
所有正式成績與 report.md 由一鍵腳本執行後產生；原始交付報告標示「尚未執行」。
Gemini 部分須可用的 GEMINI_API_KEY；沒有實際成功評估時 completion.json 保持未完成。
