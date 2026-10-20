# Week5 本機驗證紀錄

日期：2026-10-20。這是本次實際執行紀錄；重新訓練後應以新的 Output 與 completion.json 為準。

## 執行環境與時間

Windows 11、RTX 5090 31.84 GiB，Python 3.12.14，PyTorch 2.9.1+cu128、torchvision 0.24.1+cu128。
CUDA sm_120、AMP backward 與 optimizer step 實測通過，見 hardware.json。
本次完整正式主流程 exit 0，耗時 **1090.5 秒，約 18.2 分鐘**；不含事先安裝環境、下載 CUB 及資料完整性檢查的時間。
原始紀錄：logs/all_20261020_162945.log。訓練表中的 training_seconds 只計 epoch 訓練／驗證迴圈，不含下載、特徵快取、checkpoint I/O、最終測試和 BERTScore，因此不等於主流程總時間。

## 正式訓練與測試

| 模型 | 已跑 epochs | 最佳 epoch | 完整測試結果 |
| --- | --- | --- | --- |
| ResNet50 檢索 | 20 | 20 | mAP@10 0.530386，Hit@1 0.697359 |
| ViT-B/16 檢索 | 20 | 19 | mAP@10 0.658867，Hit@1 0.780751 |
| ResNet50 + DistilBERT 分類 | 20 | 17 | Accuracy 0.741974，Macro-F1 0.741493 |
| ViT-B/16 + DistilBERT 分類 | 20 | 19 | Accuracy 0.768208，Macro-F1 0.768052 |
| CLIP + Transformer | 16，patience=5 提前停止 | 11 | greedy BLEU-4 0.235462；beam BLEU-4 0.262774 |

檢索每組完整測試 2878 個 query / 2916 張 gallery；分類每組完整測試 5794 張圖片。
Caption 六組全部對齊相同 1000 張圖片、每張五參考句，沒有空描述。
CLIP beam 的 BERTScore F1 為 0.931175；Week4 Transformer beam 為 0.926703。
兩週 Transformer beam 的 BLEU-4 差值為 +0.049429，BERTScore F1 差值為 +0.004472。
這是單次 seed 的觀察，沒有做統計顯著性檢定。

## 資料完整性

- CUB 官方影像 archive MD5 通過；11788 張圖片皆配對十句有效 UTF-8 人工描述。
- 保留官方 test 5794 張，官方 train 切成 train 4795 / val 1199；各 split 均涵蓋 200 類。
- CUB 沒有跨 split 或 query/gallery 的完全相同影像。train 內 ID 3310 / 3340 是同鳥種的同一 JPEG，保留並記錄。
- Flickr8k 8000 張圖片已建立 SHA256；沒有跨 split 的完全相同圖片。train 中 3050606344_af711c726c.jpg / 2851198725_37b6027625.jpg 是相同 JPEG，沿用 Week4 切分。
- DistilBERT、CLIP 使用設定檔中的固定模型 revision；完整特徵快取含圖片 ID、資料識別與檔案 SHA256。
- Week4 原始資料、模型、預測及報告保持唯讀使用；六組 BLEU / BERTScore 在 Week5 另存。

## 程式驗證

- **13 項離線測試全部通過**：SupCon 梯度、檢索多正例指標、自身排除、圖文 epoch 配對、因果遮罩與生成、BLEU、archive 路徑防護、續傳、斷點續訓一致性、Notebook 結構，以及五模型／六組比較的整合流程。
- 五個完整架構以 batch=128 在實際 GPU 執行 AMP backward / AdamW；小型檢查峰值配置記憶體最高約 9.26 GiB，見 analysis/gpu_probe.json。這是架構檢查，不是完整訓練峰值保證。
- 真實預訓練 DistilBERT、CLIP、BERTScore 已實際載入；CLIP 自訂前處理與官方處理器在檢查影像上的最大差值為 0。
- Windows persistent workers 的 epoch 擴增可重現、圖文特徵對齊，見 analysis/worker_probe.json。
- analysis/validate_results.py 重算 CSV Top-10 的 mAP@10 / Hit@1 / Precision / Recall、分類機率與 Macro-F1／混淆矩陣，以及六組 BLEU、逐圖 BERTScore 平均；**全部通過**，見 analysis/results_audit.json。
- analysis/inference_probe.py 實際執行三個公開推論指令：檢索排除 gallery 中相同原圖；分類 Top-1 與保存結果一致；caption 單圖 beam 與批次結果一致。見 analysis/inference_probe.json。
- Python compileall、Bash 語法檢查與兩個啟動腳本 dry run 通過。

合成樣本只存在測試臨時目錄，不作為正式成績，也不會被算成 1000 張完整測試。

## 結果解讀與已知限制

分類訓練準確率接近 99%，驗證約 73–77%，有過擬合跡象；使用驗證最佳權重，不使用測試分數選 epoch。
Caption 的最佳驗證 token loss 在 epoch 11，之後未改善而提前停止，符合有限資料下的過擬合型態。

已視覺檢查檢索、分類與六張 caption 範例。CLIP beam 能描述雪地中的兩隻狗及草地跑動的狗；
但 2677656448_6b7e7702af.jpg 的游泳狗被描述成人與黃色 kayak，顯示背景干擾；
241347760_d44c8d3a01.jpg 的球員近照被加上 running 動作，顯示生成幻覺。
因此 BLEU／BERTScore 的提升不能解讀為每張圖都正確，0.931 的未 rescale BERTScore 也不是 93.1% 描述正確率。

CUB 可能與 ImageNet / Flickr 預訓練資料重疊；CLIP 與 Week4 ResNet 的預訓練資料不同，不宣稱已隔離所有預訓練因素。

第一份 log 的 bert-score 多參考句 verbose 訊息顯示約 485xxx 秒，是套件 scorer.py 將 start 計時變數覆寫成參考句索引所致，與分數無關。
本專案已停用該 verbose 計時並改用自己的 perf_counter；保留原始 log，不竄改既有紀錄。
套件內部 hash 顯示 version=0.3.12，但安裝套件為 0.3.13，原始 hash 保留以利重現。

投影片合計 95 分的全部基礎／進階項目已有實際成果，completion.json 為 true；實際評分由老師依繳交規則判定。
