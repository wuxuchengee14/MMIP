# Week5：影像檢索、多模態分類與 CLIP Image Captioning

延續 AI / Code / Data / Output 結構。涵蓋投影片的全部基礎與進階：Quiz1 30、Quiz2 35、Quiz3 30，合計 95 分。
實作五個主要模型：CNN / ViT 檢索、兩組影像+文字融合分類、CLIP+Transformer 描述。
訓練後再將 Week4 的四組描述與 Week5 的兩組描述統一重算 BLEU / BERTScore。
不需要 Gemini 或其他付費 API。

執行成果見 [REPORT.md](REPORT.md)，逐項要求見 [RUBRIC.md](RUBRIC.md)，
驗證紀錄見 [VALIDATION.md](VALIDATION.md)，繳交檔案對照見 [SUBMISSION.md](SUBMISSION.md)。

## 資料夾與報告位置

```text
week5/
├── quiz1/                  影像檢索：CNN 與 ViT
│   ├── AI/
│   ├── Code/
│   ├── Data/
│   └── Output/report.md
├── quiz2/                  影像與文字的多模態分類
│   ├── AI/
│   ├── Code/
│   ├── Data/
│   └── Output/report.md
├── quiz3/                  CLIP Image Captioning
│   ├── AI/
│   ├── Code/
│   ├── Data/
│   └── Output/report.md
├── common/                 三題共用的資料處理、訓練與報告程式
├── analysis/               結果核對與推論檢查
└── tests/                  流程測試
```

依照 Week 2 的安排，每題 `AI` 保存 AI 使用紀錄、`Code` 保存程式與 notebook、`Data` 保存資料及特徵快取、`Output` 保存圖表、指標與預測。每題完整報告固定為 `Output/report.md`。`quiz1～quiz3` 名稱也用於 Python 模組匯入，因此保留原有命名。

- [Quiz 1 報告](quiz1/Output/report.md)：CUB 資料準備、ResNet／ViT 檢索、mAP 與檢索案例。
- [Quiz 2 報告](quiz2/Output/report.md)：影像與 DistilBERT 特徵融合、兩組 Backbone 分類比較。
- [Quiz 3 報告](quiz3/Output/report.md)：CLIP Encoder 與 Transformer Decoder、Week 4／5 的 BLEU 與 BERTScore 比較。

GitHub 保留程式、設定、環境檔、資料來源說明、報告、圖表、CSV／JSON 評估結果。根目錄 `.gitignore` 排除原始資料、下載壓縮檔、模型與續訓權重、特徵／檢索向量快取、日誌及本機硬體資訊；各題 `Data/README.md` 仍可上傳。本機檔案保留，從 GitHub 重新取得專案後須下載資料並訓練模型，或自行放回權重。

## 現有 MMIP 環境

    conda activate MMIP
    python -m pip install -r week5/requirements.txt

新電腦尚無 MMIP 時才建立：

    conda env create -f week5/environment.yml
    conda activate MMIP

本機保留 PyTorch 2.9.1+cu128 / torchvision 0.24.1+cu128，支援 RTX 5090 sm_120。
本週新增 transformers 4.57.6、bert-score 0.3.13、gdown；其他相容套件可沿用。
truststore 使用 Windows 系統憑證，HTTPS 驗證保持開啟。

## 一次跑完（Git Bash）

在此儲存庫根目錄：

    conda activate MMIP
    bash week5/train_rtx5090.sh

已在 week5 目錄則使用 bash train_rtx5090.sh。
腳本優先選啟用的 Conda Python；PowerShell 的 bash.exe 可能是 WSL，請使用 Git Bash，
或改用原生 PowerShell：

    .\week5\train_rtx5090.ps1

加 --install 可先安裝 requirements；--dry-run 只顯示命令。
PowerShell 對應 -Install / -DryRun。MMIP_PYTHON、MMIP_CONFIG 可指定解譯器與設定檔。
同一時間只啟動一份流程。

## 執行順序

1. CUB 官方圖片與人工文字描述配對；保留官方影像 test，train 分層切 20% validation。
2. 每個 validation/test 鳥種類別各分 query/gallery；訓練 ResNet50、ViT-B/16 檢索模型。
3. 抽取凍結 DistilBERT 特徵；訓練 ResNet50+DistilBERT、ViT-B/16+DistilBERT 分類。
4. 快取凍結 CLIP ViT-B/32 patch 特徵；訓練 Transformer 描述模型，跑 greedy / beam。
5. 六組實際預測核對 1000 張 ID 與五參考句，再用同一設定重算 BLEU / BERTScore。
6. 產生每題 Output/report.md、根目錄 REPORT.md 與 completion.json。

程式以完成的輸出核對要求；尚未訓練時不會填入預設成績。
all / compare / report 在要求未齊全時 exit 3，執行錯誤保留 traceback 與 log。

## 資料下載與依賴

CUB 官方影像約 1.2 GB，使用四個可續傳的分段下載串流，合併後驗 MD5。Reed 人工描述透過 AttnGAN 作者提供的 birds.zip 取得，
內含 text.zip 時自動解壓；只讀原始 .txt，不載入其 pickle 預訓練嵌入。
每張必須有十句原始人工描述，缺漏會停止，不生成或補造描述。
未完成的圖片壓縮檔保留為 .part，下次在伺服器支援 Range 時接續；若伺服器拒絕 Range 則重新下載並驗證 MD5。
自動下載若被 Google Drive 限制，從原作者 https://github.com/taoxugit/AttnGAN 的 Data / birds 下載後：

    export CUB_CAPTIONS_ARCHIVE='D:/path/to/birds.zip'
    bash week5/train_rtx5090.sh --stage prepare

或 export CUB_CAPTIONS_DIR='D:/path/to/birds/text'，指向包含 200 個鳥種子目錄的 text/。
設定檔亦可填 captions_url 或 captions_directory。原始檔完整性與配對全部通過後才寫入 manifest。

Quiz3 直接唯讀使用 ../week4 的 Flickr8k 圖片、三個 JSON 分割與 vocab.json，
以及 Week4 根目錄 config.rtx5090.json、quiz4/Output/{lstm,transformer}/{greedy,beam}/predictions.json、timing.json。
這些檔案要保留；如果搬到其他位置，在 config.rtx5090.json 修改 week4_root。
不會修改 Week4 任何檔案，也不需要重新訓練 Week4。
首次準備會逐張建立 Flickr 影像 hash；之後同一來源與分割會沿用 manifest 以減少 Windows 檔案開啟成本。
若手動修改原始圖片，須先保留舊結果並移開 Flickr manifest／CLIP feature cache，再重建。

## 分段執行與續跑

    bash week5/train_rtx5090.sh --stage prepare
    bash week5/train_rtx5090.sh --stage retrieval
    bash week5/train_rtx5090.sh --stage multimodal
    bash week5/train_rtx5090.sh --stage caption
    bash week5/train_rtx5090.sh --stage compare

先看報告／更新完成清單：python week5/run_all.py --stage report。
Ctrl+C 後同一指令續跑最後完成的 epoch；best.pt 選驗證最佳、last.pt 保存 optimizer/AMP/RNG/history。
訓練設定、資料或特徵識別變更會拒絕混用舊權重；須先備份並移開對應模型的 Output 子目錄，
若變更特徵設定，也要備份並移開對應 Data 特徵快取，再重新執行。
CUDA 非嚴格確定性，固定 seed 仍不保證逐位元相同。

## RTX 5090 預設

檢索 batch=32 類×4 張=128；融合分類 batch=128；CLIP 特徵與 decoder batch=128。
預訓練 CNN/ViT learning rate=3e-5，新投影／分類層=3e-4，初始 AMP loss scale=1024。
AMP FP16、TF32、8 個 CPU threads、4 個影像 workers，預先快取 frozen 文字／CLIP 特徵。
快取特徵的 caption 訓練使用 workers=0，避免 Windows 進程重複搬運大型 Tensor。
訓練五個模型依序執行，沒有同時把全部 Backbone 放到 GPU。
實際耗時依資料下載、磁碟與 early stopping 決定，不以 GPU 型號假裝量測時間。

## 使用自己的圖片推論（先完成對應模型訓練）

    python week5/predict.py retrieval --image D:/images/bird.jpg --backbone resnet50 --top-k 10
    python week5/predict.py multimodal --image D:/images/bird.jpg --text "a bird with black wings and a yellow chest" --backbone vit_b_16
    python week5/predict.py caption --image D:/images/photo.jpg --beam-width 3

預設輸出 inference/result.json，可用 --output 指定其他檔名。檢索從固定 test gallery 搜尋，
依原圖 hash 排除查詢圖本身；多模態須同時提供圖片及文字。
推論讀取訓練時的設定與最佳權重，不再訓練，也不呼叫付費 API。

## 評估解讀

- 檢索相關性是「同鳥種」，不是辨識同一隻鳥；query/gallery 與訓練影像分開。
- hit@K 與多正例 recall@K 分開報告，mAP@K 分母明定，圖示不取代全 query 評估。
- 多模態 test 每圖只用固定第一句，train 每 epoch 選一句；同圖十句不跨 split。
- CUB 官方指出可能與 ImageNet / Flickr 預訓練資料重疊，報告保留此限制。
- CLIP 凍結，adapter 和 Decoder 實際訓練；主要 Decoder 設定與 Week4 一致，投影輸入維度不同。
- BLEU 四組 Week4 + 兩組 Week5 全部重算；BERTScore 記錄 model/layer/hash、五參考句最大值規則及逐圖分數。
- 時間表使用同圖片和相同條件；pipeline 均值之和只是估計，不是逐張端到端量測。

離線測試：python -m unittest discover -s week5/tests -v。
GPU 小型架構檢查：python week5/tests/gpu_probe.py。
正式結果核對：python week5/analysis/validate_results.py（須先完成全部正式輸出）。
測試使用的合成樣本只測流程，不能當作作業成績。
