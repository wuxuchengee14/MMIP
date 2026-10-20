# AI 協助使用說明

本題由 Codex 協助撰寫 CLIP 特徵抽取、Transformer 訓練、描述生成及跨週評估。
本週 Transformer Decoder 與投影層有實際訓練，CLIP 影像編碼器使用凍結預訓練權重。
BLEU 與 BERTScore 使用保存的真實生成句及相同五參考句計算；無需 Gemini 或其他外部評分 API。
提交前應閱讀 Output/report.md 並檢視生成範例，不能把 BLEU／BERTScore 解讀為描述正確率。
