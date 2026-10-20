# 資料、模型與方法來源

- CUB-200-2011：200 鳥種、11788 張影像，官方圖片標籤／train-test 分割。
  https://www.vision.caltech.edu/datasets/cub_200_2011/
  https://data.caltech.edu/records/65de6-vp158
  原始 archive MD5：97eceeb196236b17998738112f37df78。
  官方提醒 ImageNet/Flickr 預訓練資料可能與 CUB 重疊；本作業不宣稱排除此限制。
- Reed et al., Learning Deep Representations of Fine-Grained Visual Descriptions, CVPR 2016：每圖十句人工描述。
  https://arxiv.org/abs/1605.05395
  https://github.com/reedscot/cvpr2016
  使用 AttnGAN 作者公開分發的原始文字： https://github.com/taoxugit/AttnGAN
  birds.zip 中的 text.zip 解壓後逐張核對十句；不執行／載入其中的 pickle 或舊版訓練程式。
- Flickr8k：沿用本機 Week4 已準備的官方 image-level split、五段人工參考句與原始詞彙表。
  下載鏡像與 Week4 來源記錄見 ../week4/quiz1/Data/flickr8k/manifest.json。
- ResNet50 / ViT-B/16：torchvision ImageNet-1K V1 預訓練。
  https://docs.pytorch.org/vision/stable/models.html
- DistilBERT： https://huggingface.co/distilbert/distilbert-base-uncased
- CLIP ViT-B/32： https://huggingface.co/openai/clip-vit-base-patch32
  https://huggingface.co/docs/transformers/model_doc/clip
- Supervised Contrastive Learning： https://arxiv.org/abs/2004.11362
- BLEU： https://aclanthology.org/P02-1040/
  程式與 tokenization 由本專案 Week4 實作沿用，固定多參考句、無 smoothing。
- BERTScore： https://github.com/Tiiiger/bert_score
  固定 FacebookAI/roberta-base 第十層、無 IDF、不 rescale，保存 evaluator hash；
  套件 0.3.13 的內部 scorer hash 可能顯示 version=0.3.12，保留原字串以利核對，不自行改寫。

實作參考公開方法／API 文件，模型成績必須來自本機實際訓練和相同測試集評估，不使用論文數字代填。
