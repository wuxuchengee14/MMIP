# Quiz 1：資料集準備

三種資料均使用固定、互斥的訓練／驗證／測試集合。

| dataset | train | val | test |
| --- | --- | --- | --- |
| imdb | 20000 | 5000 | 25000 |
| organamnist | 34561 | 6491 | 17778 |
| flickr8k | 6000 | 1000 | 1000 |

## 資料來源與處理

- IMDb: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz；官方訓練集分層切出 20% validation；測試集保持不變。移除 HTML、轉小寫、以正規表示式斷詞，詞彙表只由 training 建立。
- OrganAMNIST: https://medmnist.com/v2；11 類器官 CT，原始 28×28 灰階。官方 train/val/test 分割；複製既有原始檔並檢查官方 MD5，不沿用第三週模型。
- Flickr8k: 使用 TensorFlow 官方教學所引用的下載鏡像；依官方影像清單分割，每張圖的五段描述不跨集合。詞彙表只由 training captions 建立。
- Flickr8k 鏡像來源與說明：https://www.tensorflow.org/text/tutorials/image_captioning

## 限制

OrganAMNIST 放大不會增加細節；這裡沒有病患 ID，無法另行驗證病患層級重疊。測試集只用於最終評估，不用來挑選權重或建立詞彙表。

類別分布見 organ_class_counts.csv，器官範例見 organ_examples.png。文字範例見 datasets.json，圖片與五段描述可由 Data/flickr8k/train.json 查閱。
