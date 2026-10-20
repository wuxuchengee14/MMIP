# 圖文分類資料

圖片、標籤及原始人工描述共用 ../../quiz1/Data/cub/，不複製大型影像。
text_features/ 保存凍結 DistilBERT 的特徵與圖片 ID，complete.json 保存模型 revision 與資料識別。
訓練有十句特徵／圖，每個 epoch 選一句；驗證與測試固定使用第一句。
快取是衍生資料，可在保留舊結果後重新抽取；原始人工文字仍在 Quiz1 的 captions/。
