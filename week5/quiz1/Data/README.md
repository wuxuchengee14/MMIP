# CUB 資料

執行根目錄流程的 prepare 階段，會在 cub/ 下載、驗證並整理資料。
影像官方壓縮檔驗 MD5；每張影像配對十句 Reed 人工描述，保存圖片 SHA256。

- train.json / val.json / test.json：圖片 ID、路徑、類別、十句原始描述。
- classes.json：200 個鳥種類別，索引從 0 開始。
- retrieval_splits.json：各鳥種的驗證／測試 query 與 gallery。
- manifest.json：資料識別、切分與來源。

官方 train 5994 張切為 train 4795、val 1199；官方 test 5794 張保持不變。
詳細完整性檢查見 ../Output/data_audit.json，來源及使用限制見根目錄 DATA_SOURCES.md。
