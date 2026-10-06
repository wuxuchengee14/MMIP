# Quiz 1：OrganAMNIST 資料準備

輸入為 28×28 腹部 CT 灰階軸向器官影像，輸出為 11 個器官類別之一。
來源：MedMNIST / LiTS；授權：CC BY 4.0。

官方 train=34561、validation=6491、test=17778。
合併 train 與 validation 共 41052 張，採用 5-fold，seed=42。
切分方法：StratifiedKFold (image level)。Test 保留官方切分，未參與 fold、超參數選擇或 early stopping。

標準 NPZ 無病例 ID，因此無法保證 CV 病例獨立，應以官方 test 評估泛化。
開發集重複影像數：0；
開發集與官方 test 共用的不同影像雜湊數：0。
若存在官方跨集重複，保留官方測試集並揭露此限制。

類別數量、影像範例、fold 分布與原始索引對應見同資料夾 CSV / PNG。
