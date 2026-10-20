# Caption 資料

flickr_manifest.json 記錄唯讀使用的 Week4 Flickr8k 路徑、切分、詞彙表及圖片雜湊。
來源由根目錄 config.rtx5090.json 的 week4_root 指定。
每張圖五段人工參考句；圖片切分 6000 / 1000 / 1000，詞彙表沿用 Week4。
clip_features/ 保存凍結 CLIP ViT-B/32 的 4×4 空間特徵，以及確切模型 revision／資料識別。
不得將測試參考句輸入描述生成模型；參考句只用於事後評估。
