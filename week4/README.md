# Week 4：RNN、LSTM、ViT 與 Image Captioning

涵蓋 Quiz 1～4 全部基礎與進階要求，共 100 分。依參考專案採用每題 AI / Code / Data / Output 格式：
https://github.com/wuxuchengee14/MMIP
保留本機已建立的 quiz1～quiz4 名稱。所有路徑由程式位置解析，可從專案根目錄或本週目錄執行。

## 資料夾結構

```text
week4/
├── quiz1/
│   ├── AI/
│   ├── Code/
│   ├── Data/
│   └── Output/report.md
├── quiz2/
│   ├── AI/
│   ├── Code/
│   ├── Data/
│   └── Output/report.md
├── quiz3/
│   ├── AI/
│   ├── Code/
│   ├── Data/
│   └── Output/report.md
└── quiz4/
    ├── AI/
    ├── Code/
    ├── Data/
    └── Output/report.md
```

每題的 `AI` 保存 AI 使用紀錄，`Code` 保存程式與 notebook，`Data` 保存原始資料或共用資料說明，`Output` 保存模型評估、圖表、預測結果與報告。原始資料、模型權重、訓練日誌和本機硬體資訊由根目錄 `.gitignore` 排除，不會上傳 GitHub。

## 環境

目前 MMIP 的 Python 3.12、torch 2.9.1+cu128 / torchvision 0.24.1+cu128 可沿用。
新環境使用 conda env create；已有同名環境只需 pip 安裝依賴，勿重建覆蓋。

    conda env create -f week4/environment.yml
    conda activate MMIP

已有 MMIP：

    conda activate MMIP
    python -m pip install -r week4/requirements.txt
    python week4/check_gpu.py

所有模型使用 PyTorch；BLEU 與 Gemini REST 使用專案實作／Python 標準函式庫，
不需 TensorFlow、torchtext、NLTK 語料下載或 Google SDK。
truststore 使用 Windows 系統憑證修正 Conda 的下載憑證問題，維持 HTTPS 驗證。

## 一次跑完：Windows Git Bash

在 Git Bash（不是 PowerShell 的 WSL bash.exe）執行：

    conda activate MMIP
    export GEMINI_API_KEY="你的_API_key"
    bash week4/train_rtx5090.sh

若已進入 week4，執行 bash train_rtx5090.sh。
腳本優先使用啟用的 Conda Python，不以 PATH 中其他 Python 取代。
可指定 MMIP_PYTHON；GEMINI_MODEL 可覆寫設定檔的 Gemini 模型。
金鑰只從環境讀取，不寫入設定、報告或 log。

PowerShell：

    conda activate MMIP
    $env:GEMINI_API_KEY = "你的_API_key"
    .\week4\train_rtx5090.ps1

只檢查命令、不開始執行：

    bash week4/train_rtx5090.sh --dry-run

安裝套件後執行：加 --install；它會使用同一個 Python 安裝 requirements.txt。

## 執行順序

1. IMDb、OrganAMNIST、Flickr8k 下載、檢查與固定分割。
2. RNN / LSTM：各一次訓練，測試集推論、Accuracy / F1 / ROC-AUC。
3. ViT-B/16 / ResNet-50：各一次 head warmup 與全模型微調，Top-1 / Top-5 / Macro-AUC。
4. 共用凍結 ResNet 影像特徵，訓練 LSTM / Transformer caption decoder。
5. 兩個 caption 模型於全部測試圖片生成 greedy / beam 描述，計算 BLEU-1～4、測量時間。
6. Gemini 對相同 100 張固定測試圖片的兩個 greedy 描述各作一次評估，共 200 筆成功結果。
7. 更新各題 Output/report.md、根目錄 REPORT.md 與 completion.json。

六個主要模型；ViT 與 ResNet 各另有一個短暫 head warmup。
所有結果由實際訓練／推論產生，原始碼交付時不包含預先編造的成績。

## Gemini 與完成狀態

實際呼叫會把抽樣 Flickr8k 測試圖片與生成描述送至 Google Gemini，可能消耗 API 配額／費用。
採目前官方列出的 gemini-3.8-flash，可透過 GEMINI_MODEL 換成帳號可用且支援圖片與 JSON 的模型。
參考：https://ai.google.dev/api/generate-content 和 https://ai.google.dev/gemini-api/docs/models
同一張圖／描述／評估模型／prompt 會快取成功回應，重新執行不重複呼叫。
HTTP 429/5xx 會有限次退避重試；認證／模型／配額錯誤保留待辦，不能視為完成。

沒有 key 仍會完成本機訓練，最後回傳 exit code 3，明確標示 Gemini 待完成。
若暫時只跑本機可用 --skip-gemini，同樣不會宣稱全部要求已完成。
取得 key 後只補 Gemini，不重訓：

    export GEMINI_API_KEY="你的_API_key"
    export GEMINI_INTERVAL_SECONDS=15
    bash week4/train_rtx5090.sh --stage gemini

GEMINI_INTERVAL_SECONDS 設定每次請求結束後至少間隔幾秒（包括失敗後的請求），
15 秒是保守起點，仍須依 AI Studio 同一專案／模型的 RPM、TPM、每日配額調整。
同一時間只執行一份流程；其他使用同專案的程式也可能共用配額。
此環境變數只影響 API 等待，不變更訓練設定或成功快取的識別。

429／503 等暫時錯誤預設最多重試 5 次，採 10、20、40、80、120 秒加隨機抖動的退避，
並遵守 Retry-After／RetryInfo 指定的較長等待時間。
若 API 明確指出每日配額用盡、配額為零，或建議等待超過 300 秒，會停止並保留待辦；
持續失敗到重試上限也會停止本輪，不再對其餘圖片連續送出失敗請求。
若是每日配額，縮短圖片或降低頻率都不能增加可用請求數，須等待重設或調整配額。
503 持續發生則稍後再續跑；換模型會產生不同評估條件，現有結果不能混入新模型的比較。

quiz4/Output/gemini/errors.json 記錄 HTTP 狀態、安全的配額識別與建議等待秒數，
不保存完整錯誤訊息、請求標頭或金鑰。成功／失敗後即更新 summary.json，
只有兩個模型各 100 筆全部成功才會標記 complete=true。
既有舊紀錄沒有配額細節，無法事後還原，須由下次錯誤回應或 AI Studio 確認。

## 分段與中斷續訓

Gemini API 免費額度不足時，也可手動上傳圖片至 Gemini 網頁：

    python week4/quiz4/Code/manual_gemini.py --prepare

依 quiz4/Output/gemini_manual/README.md 操作：20 批、每批 5 張圖片，每張 A/B 兩段匿名描述。
保存每批原始 JSON 回覆至 responses/，填寫 session.json，再執行：

    python week4/quiz4/Code/manual_gemini.py --import-all

通過完整性與格式檢查後會更新比較報告及 completion.json；未滿 200 筆不標記完成。
網頁手動評估與既有 API 評估分開保存、分開統計，報告明示執行方式。

依使用者指定縮為 10 張代表案例展示時，使用獨立資料夾 gemini_cases10：

    python week4/quiz4/Code/manual_gemini.py --case-study --prepare
    python week4/quiz4/Code/manual_gemini.py --case-study --import-all

共兩批，每批五張，兩模型合計 20 段描述；按該資料夾 README.md 操作。
全部 20 筆驗證後可完成案例展示形式的 Gemini 要求，報告會明示選例方式與範圍，
不能宣稱原 100 張／200 筆研究或 API 全部完成，也不能把挑選案例的符合率外推全體資料。

### 本機訓練階段

    bash week4/train_rtx5090.sh --stage prepare
    bash week4/train_rtx5090.sh --stage sentiment
    bash week4/train_rtx5090.sh --stage vision
    bash week4/train_rtx5090.sh --stage caption

Ctrl+C 後重新執行相同指令會從最後完成的 epoch 接續，完成的模型會載入 best.pt。
last.pt 保存模型、optimizer、AMP scaler、RNG、early stopping 和 history。
訓練設定／資料識別變更會拒絕混用 checkpoint。不要同時啟動兩份流程。
CUDA 使用非嚴格確定性高效 kernel，因此固定 seed 不保證 GPU 每次逐位元一致。

## RTX 5090 設定與限制

設定檔 config.rtx5090.json：文字 batch 128、影像 batch 128、caption batch 128。
AMP FP16、TF32；影像載入使用 4 workers，預先快取 caption 特徵。
器官分類只傳輸原始 28×28 uint8，整批在 GPU 放大／正規化，避免 CPU 逐張放大與大筆傳輸。
特徵已在記憶體時用 workers=0，避免 Windows 複製與 IPC 開銷。
GPU 不足時降低對應 batch_size；已開始的訓練若改設定需先另存舊結果並以新目錄重跑。

5090 可加速 GPU 計算，但首次下載、磁碟讀取、逐字生成與 Gemini 配額等待仍會影響總時間。
程式記錄真實 epoch／推論／API 時間，沒有以 GPU 型號推估成績或完成時間。
建議保留至少 20 GB 可用磁碟空間（原始資料、特徵、模型及 optimizer checkpoint）。

## 資料與公平比較

- IMDb：官方 25,000 train 切成 20,000 train / 5,000 validation，25,000 test 保持不變。
- OrganAMNIST：官方 34,561 / 6,491 / 17,778；第三週原始檔可直接複製並驗 MD5。
- Flickr8k：官方 6,000 / 1,000 / 1,000 張影像；每張五段描述，同圖不跨集合。
- 詞彙表只由 training 建立；模型依 validation 指標選權重，test 不用於挑選。
- 影像分類兩模型共用確定性前處理、不擴增；caption 兩模型共用相同凍結編碼器。
- 本週未要求 K-fold，採單一固定分割完成全題，沒有額外的五折成本。

## 成果位置

- quiz1/Output/report.md：三資料集、來源、統計、範例與分割。
- quiz2/Output/report.md：RNN / LSTM 結果與案例。
- quiz3/Output/report.md：ViT / ResNet 結果與 Macro-AUC。
- quiz4/Output/report.md：Captioning、BLEU、時間與 Gemini 評估。
- 各模型子目錄：best.pt / last.pt / history.csv / complete.json。
- logs/：每次執行 log；completion.json：全部要求是否完成。
- 各題 Code/quizN.ipynb：讀取報告與結果，訓練預設關閉。

輸出與大型資料預設不納入 Git；繳交時另外打包需要的成果。
分享報告時應一併帶上所引用的圖片／CSV／JSON；模型要推論則須保留詞彙表、設定與 best.pt。
