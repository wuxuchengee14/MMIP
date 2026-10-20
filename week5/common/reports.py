from common.utils import ROOT,read_json,report,table,write_json


def compact(rows, keys):
    return table([{k:r[k] for k in keys if k in r} for r in rows])


def comparison_analysis(quiz, rows):
    """Describe observed differences only; preserve all raw columns in comparison.csv."""
    if not rows:
        return ""
    if quiz == "quiz1":
        key, metric = "backbone", "map_at_10"
        metrics = [("map_at_10", "mAP@10"), ("hit_at_1", "Hit@1")]
        winner = max(rows, key=lambda r:r[metric])
        other = min(rows, key=lambda r:r[metric])
        text = (f"本次 {winner[key]} 的 mAP@10 較高（{winner[metric]:.4f}），"
                f"與另一組相差 {100*(winner[metric]-other[metric]):.2f} 個百分點。"
                "這是同鳥種圖片的排序品質，不能解讀為鳥種分類準確率。")
    elif quiz == "quiz2":
        key, metric = "image_backbone", "accuracy"
        metrics = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1")]
        winner = max(rows, key=lambda r:r[metric])
        other = min(rows, key=lambda r:r[metric])
        text = (f"本次 {winner[key]} + DistilBERT 的測試 Accuracy 為 {winner[metric]:.4f}，"
                f"較另一組高 {100*(winner[metric]-other[metric]):.2f} 個百分點；"
                f"其 Macro-F1 為 {winner['macro_f1']:.4f}。"
                "Macro-F1 讓每個鳥種等權重，可與 Accuracy 一起檢查少數類別的表現。")
        import csv
        gaps=[]
        for r in rows:
            directory=ROOT/quiz/"Output"/r[key]
            done=optional(str((directory/"complete.json").relative_to(ROOT)))
            if done and (directory/"history.csv").exists():
                with (directory/"history.csv").open(encoding="utf-8-sig",newline="") as stream:
                    history=list(csv.DictReader(stream))
                best=next(h for h in history if int(h['epoch'])==done['best_epoch'])
                train,val=float(best['train_accuracy']),float(best['val_accuracy'])
                gaps.append(f"{r[key]} 最佳 epoch {done['best_epoch']} 的 train / val Accuracy 為 {train:.4f} / {val:.4f}")
        if gaps:
            text+=" "+"；".join(gaps)+"。"
            text+=" 訓練與驗證的差距需配合 learning_curves.png 判讀過擬合；訓練影像含隨機擴增、文字也會輪換，兩者測量條件不同。"
    else:
        key = "method"
        rows = [dict(r,method=f"{r['week']} {r['model']} {r['decoding']}") for r in rows]
        metrics = [("bleu4", "BLEU-4"), ("bertscore_f1", "BERTScore F1")]
        best_bleu=max(rows,key=lambda r:r['bleu4'])
        best_bert=max(rows,key=lambda r:r['bertscore_f1'])
        text=(f"六組中 BLEU-4 最高為 {best_bleu[key]}（{best_bleu['bleu4']:.4f}）；"
              f"BERTScore F1 最高為 {best_bert[key]}（{best_bert['bertscore_f1']:.4f}）。")
        old=[r for r in rows if r['week']=='week4' and r['model']=='transformer']
        new=[r for r in rows if r['week']=='week5']
        for r in new:
            baseline=next((b for b in old if b['decoding']==r['decoding']),None)
            if baseline:
                text+=(f" CLIP {r['decoding']} 相對 Week4 Transformer 的 BLEU-4 差值為 "
                       f"{r['bleu4']-baseline['bleu4']:+.4f}，BERTScore F1 差值為 "
                       f"{r['bertscore_f1']-baseline['bertscore_f1']:+.4f}。")
        text+=" 指標升降是本次固定測試集上的觀察，較高的語意相似度仍可能伴隨物體或動作描述錯誤。"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    labels=[r[key] for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(12, max(3.3,len(rows)*.65)))
    y=np.arange(len(rows))
    for ax,(field,title) in zip(axes,metrics):
        bars=ax.barh(y,[r[field] for r in rows],color="steelblue")
        ax.set_yticks(y,labels,fontsize=8);ax.invert_yaxis()
        ax.set_xlim(0,1);ax.set_title(title)
        ax.bar_label(bars,fmt="%.3f",padding=3,fontsize=8)
        ax.set_xlabel("Score (0–1)")
    fig.tight_layout();fig.savefig(ROOT/quiz/"Output/comparison.png",dpi=150);plt.close(fig)
    return "\n\n## 結果分析\n\n"+text+"\n\n![完整測試集指標比較](comparison.png)\n\n完整欄位保留於 comparison.csv / comparison.json。"


def optional(relative):
    path=ROOT/relative
    return read_json(path) if path.exists() else None


def make_reports(config):
    data=optional("quiz1/Data/cub/manifest.json")
    rows=optional("quiz1/Output/comparison.json")
    report("quiz1","Quiz 1：CNN / ViT 鳥類影像檢索",
        "## 資料與切分\n\nCUB-200-2011 影像與鳥種類別；Reed 原始人工描述另供 Quiz2 使用。"
        "官方 image-level test 保持不變，官方 train 分層切出 validation。驗證與測試各自按鳥種拆 query/gallery，"
        "兩者不含同一 image ID；以同鳥種定義相關結果，這是類別語意檢索，不是同一隻鳥的個體識別。\n\n"+
        (table([{"split":s,"images":n} for s,n in data["counts"].items()]) if data else "**待準備真實資料。**")+
        ("\n\n![CUB 圖文樣本](cub_examples.png)\n\n資料檢查見 data_audit.json；訓練集有一組同類別重複 JPEG，"
         "沒有跨 train/val/test 或 query/gallery 的完全相同影像。" if (ROOT/"quiz1/Output/data_audit.json").exists() else "")+
        "\n\n## 方法\n\nResNet-50 / ViT-B/16 使用 ImageNet-1K V1 預訓練，整個影像編碼器與 256 維投影層實際訓練。"
        "P 個鳥種、每類 K 張不同圖片的 batch，loss = supervised contrastive loss + 0.5 × CE；"
        "L2-normalized embedding 以 cosine similarity 排序。train 隨機裁切與翻轉，val/test 確定性 resize/center crop。"
        "依 validation mAP@10 選模型，test 不用於選權重。history 的 train_accuracy 是輔助分類 Top-1，"
        "val_accuracy 欄位對應選模用的 mAP@10，兩者意義不同。\n\n## 全部測試 query 結果\n\n"+
        (compact(rows,("backbone","map","map_at_10","hit_at_1","recall_at_10","training_seconds","parameters")) if rows else "**尚未完成正式訓練與測試，沒有填入成績。**")+
        comparison_analysis("quiz1",rows)+
        "\n\nhit_at_K：前 K 張至少有一張相關圖片的 query 比率；recall_at_K：前 K 張找回的相關圖片數／"
        "gallery 全部相關圖片數，再對 query 平均。precision_at_K 為前 K 張相關比例；"
        "AP@K 分母 min(K, gallery 相關圖片數)，mAP 是各 query AP 的平均。map 欄位評估完整排序。"
        "所有 query 都納入指標，示意圖挑選較強與較弱案例，綠色為同類、紅色為不同類。\n\n"
        "每模型附 best.pt、history.csv、learning_curves.png、metrics.json、embeddings.npz、"
        "retrieval_results.csv、retrieval_examples.png。\n\n## 限制\n\n"
        "CUB 官方提醒資料可能與 ImageNet/Flickr 預訓練資料重疊，因此不宣稱對預訓練影像完全獨立的泛化結果。"
        "兩種 Backbone 共用相同資料、設定與評估條件，單次 seed 不宣稱統計顯著優勢。\n\n"
        "來源：https://www.vision.caltech.edu/datasets/cub_200_2011/；https://github.com/taoxugit/AttnGAN。")
    rows=optional("quiz2/Output/comparison.json")
    report("quiz2","Quiz 2：鳥類影像與文字的多模態分類",
        "## 資料\n\n共用 Quiz1 的 CUB 圖片 ID 與 200 類標籤、每張十句 Reed 人工描述。"
        "以圖片為切分單位，同圖的描述全部留在同一 split；不輸入標籤或含類別名稱的檔案路徑。"
        "訓練每個 epoch 每張圖選一句描述；validation/test 固定使用第一句，每張圖只評估一次。"
        "原文依檔名配對，不以生成描述或類別名稱冒充人工描述。\n\n## 兩種組合\n\n"
        "ResNet-50 + DistilBERT、ViT-B/16 + 相同 DistilBERT。文字模型凍結、attention-masked mean pooling，"
        "快取所有訓練描述。兩模態各投影至 256 維，串接後經 MLP 做分類；"
        "影像 Backbone、兩個投影層及融合分類頭實際訓練，驗證 Accuracy 選權重。\n\n"
        "## 測試比較\n\n"+(compact(rows,("image_backbone","accuracy","macro_f1","weighted_f1","training_seconds","trainable_parameters")) if rows else "**待正式訓練與完整測試，沒有預設成績。**")+
        comparison_analysis("quiz2",rows)+
        "\n\n兩組保留相同文字 Backbone、資料與融合結構，主要比較影像 Backbone。"
        "training_seconds 為 epoch 訓練／驗證迴圈，文字特徵抽取時間另存；test_seconds 使用預先快取文字特徵，"
        "不代表從原始文字開始的端到端推論延遲。\n\n"
        "每模型附 best.pt、history.csv、learning_curves.png、metrics.json、classification_report.json、"
        "confusion_matrix.png/npy、confusions.csv、predictions.csv、probabilities.npz、prediction_examples.png。"
        "完整矩陣以類別索引顯示，classes.json 可還原類名。\n\n"
        "同樣保留 CUB 預訓練重疊限制；不以融合模型分數直接主張文字或圖片各自的因果貢獻。")
    rows=optional("quiz3/Output/comparison.json")
    clip=optional("quiz3/Output/clip_results.json")
    report("quiz3","Quiz 3：CLIP Image Captioning 與 Week4 比較",
        "## 資料與模型\n\n直接讀取 Week4 Flickr8k 的 6000/1000/1000 image-level split、"
        "原始圖片、五段參考句及 training-only 詞彙表；不改動 Week4 資料、權重、預測或報告。"
        "CLIP ViT-B/32 影像編碼器凍結，移除 CLS、逐 patch post-LayerNorm，再平均池化成 4×4 空間特徵。"
        "可訓練投影層把 768 維 CLIP 特徵接上與 Week4 相同主要設定的 Transformer Decoder。"
        "teacher forcing、PAD 忽略交叉熵，validation token loss 選模型。"
        "測試從 BOS 開始生成，不提供參考句；同時跑 greedy/beam。\n\n"+
        ("影像 SHA256 檢查沒有跨 split 的完全相同圖片；訓練集中有兩個檔名對應相同 JPEG，"
         "保留 Week4 原始切分以維持比較條件，詳見 data_audit.json。\n\n" if (ROOT/"quiz3/Output/data_audit.json").exists() else "")+
        "投影輸入由 ResNet 的 2048 維改成 CLIP 的 768 維，這部分參數量不同，"
        "並非聲稱所有參數與預訓練資料都完全一致。\n\n## 本週 BLEU 與時間\n\n"+
        (compact(clip,("decoding","bleu1","bleu2","bleu3","bleu4","decoder_mean_ms","estimated_pipeline_mean_ms","training_seconds")) if clip else "**待正式訓練與生成。**")+
        "\n\n## Week4 / Week5 統一重算比較\n\n"+
        (compact(rows,("week","model","decoding","bleu4","bertscore_f1","decoder_mean_ms","estimated_pipeline_mean_ms","empty_captions")) if rows else "**待使用實際預測重算 BLEU 與 BERTScore。**")+
        comparison_analysis("quiz3",rows)+
        "\n\n比較六組：Week4 ResNet+LSTM / ResNet+Transformer，各 greedy/beam；"
        "Week5 CLIP+Transformer，greedy/beam。先核對每組完整 1000 張 ID 與五參考句，再用同一實作重算 BLEU-1～4。"
        "BERTScore 共用 RoBERTa-base 第 10 層、無 IDF、不 rescale；五參考句模式依官方函式分別取 P/R/F1 最大值，"
        "若模型生成空描述，保留該圖片並記錄 empty_captions，不從測試集刪除。"
        "保存 scorer hash、設定及逐圖分數。兩種指標不是描述正確率，預訓練評估模型也有其偏好。\n\n"
        "延遲測量共用固定前三十張測試圖、batch=1、三次重複、暖機與 CUDA 同步；"
        "估計 pipeline 時間為分開量測的 encoder 與 decoder 均值之和。特徵抽取、訓練與 BERTScore 耗時分開。"
        "不同模型可能提早停止，總訓練時間不等於相同 epoch 的效率比較。\n\n"
        "來源：https://huggingface.co/openai/clip-vit-base-patch32；https://github.com/Tiiiger/bert_score。")


def completion(config):
    checks=[]
    files={
        "CUB image/text data and Flickr manifests":["quiz1/Data/cub/manifest.json","quiz1/Data/cub/retrieval_splits.json","quiz3/Data/flickr_manifest.json"],
        "CNN and ViT retrieval":["quiz1/Output/comparison.json"]+
            [f"quiz1/Output/{k}/{p}" for k in config["retrieval"]["backbones"] for p in
             ("best.pt","complete.json","history.csv","metrics.json","retrieval_results.csv","retrieval_examples.png")],
        "Two multimodal backbone combinations":["quiz2/Output/comparison.json"]+
            [f"quiz2/Output/{k}/{p}" for k in config["multimodal"]["backbones"] for p in
             ("best.pt","complete.json","history.csv","metrics.json","predictions.csv","prediction_examples.png")],
        "Frozen CLIP + trained decoder captioning":["quiz3/Output/clip_transformer/best.pt","quiz3/Output/clip_transformer/complete.json",
                                                   "quiz3/Output/clip_transformer/history.csv","quiz3/Output/clip_results.json"]+
            [f"quiz3/Output/clip_transformer/{m}/{p}" for m in ("greedy","beam") for p in
             ("predictions.json","bleu.json","timing.json","caption_examples.png")],
        "Week4/Week5 BLEU and BERTScore comparison":["quiz3/Output/comparison.json"]+
            [f"quiz3/Output/bertscore/{w}_{k}_{m}/{p}" for w,k in (("week4","lstm"),("week4","transformer"),("week5","clip_transformer"))
             for m in ("greedy","beam") for p in ("metrics.json","per_image.npz")]
    }
    for label,paths in files.items():
        missing=[p for p in paths if not (ROOT/p).is_file()]
        checks.append({"requirement":label,"complete":not missing,"missing":missing})
    compared=optional("quiz3/Output/comparison.json")
    if compared is not None and (len(compared)!=6 or any(r.get("test_images")!=1000 for r in compared)):
        checks[-1]["complete"]=False;checks[-1]["missing"].append("Need all six methods on the exact 1000-image test set")
    done=all(c["complete"] for c in checks)
    write_json(ROOT/"completion.json",{"all_requirements_complete":done,"rubric_points_in_provided_slides":95,"checks":checks})
    (ROOT/"REPORT.md").write_text("# Week5 執行結果\n\n"+
        ("投影片列出的全部基礎／進階項目已產生實際成果。" if done else "**尚有項目未完成，見 completion.json。程式驗證通過不等同正式訓練完成。**")+
        "\n\n投影片合計 30+35+30=95 分；不自行假設其餘 5 分的要求。\n\n"+
        table([{"requirement":c["requirement"],"complete":c["complete"],"missing_count":len(c["missing"])} for c in checks])+
        "\n\n- [Quiz1](quiz1/Output/report.md)\n- [Quiz2](quiz2/Output/report.md)\n- [Quiz3](quiz3/Output/report.md)\n",encoding="utf-8")
    return done
