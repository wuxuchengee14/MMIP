"""Build reviewable submission and optional best-model archives from completed real outputs."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from common.utils import ROOT, file_hash, read_json, table, write_json

MODELS = [("quiz2", "rnn"), ("quiz2", "lstm"), ("quiz3", "vit_b_16"),
          ("quiz3", "resnet50"), ("quiz4", "lstm"), ("quiz4", "transformer")]


def submission_report():
    cases = read_json(ROOT / "quiz4/Output/gemini_cases10/summary.json")
    datasets = read_json(ROOT / "quiz1/Output/datasets.json")
    config = read_json(ROOT / "config.rtx5090.json")
    text_rows, vision_rows = [], []
    for quiz, kind in MODELS[:4]:
        folder = ROOT / quiz / "Output" / kind
        metrics, done = read_json(folder / "metrics.json"), read_json(folder / "complete.json")
        row = {"model": kind, "Accuracy (%)": round(metrics["top1"] * 100, 4),
               "Macro-F1": round(metrics["macro_f1"], 6),
               "parameters": done["parameters"], "training_seconds": round(done["training_seconds"], 2)}
        if quiz == "quiz2":
            # The sentiment comparison reports positive-class ROC-AUC.
            import csv
            with (ROOT / "quiz2/Output/comparison.csv").open(encoding="utf-8-sig", newline="") as f:
                compared = {r["model"]: r for r in csv.DictReader(f)}
            row["positive ROC-AUC"] = round(float(compared[kind]["roc_auc"]), 6)
            text_rows.append(row)
        else:
            row["Macro-AUC"] = round(metrics["macro_auc"], 6)
            row["Top-5 (%)"] = round(metrics["top5"] * 100, 4)
            vision_rows.append(row)
    caption = [{"model": r["model"], "decoding": r["decoding"],
                "BLEU-4 x100": round(r["bleu4"] * 100, 4),
                "decoder_mean_ms": round(r["decoder_mean_ms"], 3),
                "estimated_pipeline_ms": round(r["estimated_pipeline_mean_ms"], 3)}
               for r in read_json(ROOT / "quiz4/Output/comparison.json")]
    judgments = read_json(ROOT / "quiz4/Output/gemini_cases10/judgments.json")
    mapping = {(r["image_id"], r["caption_model"]): r for r in judgments}
    from quiz4.Code.manual_gemini import CASE_STUDY_SELECTION
    examples = []
    for i, (sid, reason) in enumerate(CASE_STUDY_SELECTION, 1):
        a, b = mapping[(sid, "lstm")], mapping[(sid, "transformer")]
        examples.append({"case": f"{i:02d}", "topic": reason,
            "LSTM score/verdict": f"{a['judgment']['score']}/5 {a['judgment']['verdict']}",
            "Transformer score/verdict": f"{b['judgment']['score']}/5 {b['judgment']['verdict']}"})
    return (
        "# Week 4 作業繳交報告\n\n" +
        "學生／操作人：" + cases["session"]["operator"] +
        "。本次完成 Quiz 1～4 基礎與進階項目；六模型有實際訓練、測試與輸出。\n\n"
        "Gemini 項目採使用者指定的十張代表案例，兩模型共二十筆網頁手動評估。"
        "此處完成的是案例展示範圍，沒有宣稱原 100 張 API 研究已完成。\n\n"
        "## Quiz 1：三種資料集準備\n\n" +
        table([{"dataset": name, **m["counts"]} for name, m in datasets.items()]) +
        "\n\nIMDb 為正／負情緒二分類，官方 train 分層切出 validation、官方 test 保持不變；"
        "文字去除 HTML、轉小寫、規則斷詞，字典只由 training 建立，最多前 256 tokens。"
        "OrganAMNIST 為十一類器官 CT，使用官方分割；28×28 灰階轉三通道並放大至 224×224。"
        "Flickr8k 使用官方影像層級分割，每張五段參考描述，同圖不跨集合，詞彙表只由 training captions 建立。\n\n"
        "資料來源、類別、hash、處理與樣本見 [Quiz 1 報告](quiz1/Output/report.md)、"
        "[資料資訊](quiz1/Output/datasets.json)。\n\n"
        "![OrganAMNIST 範例](quiz1/Output/organ_examples.png)\n\n"
        "IMDb split ID 互斥，但來源含少量完全相同評論。排除測試集中與 train/validation 相同的 123 篇後，"
        "LSTM Accuracy 由 83.376% 降至約 83.318%，結論差異很小；仍保留原官方測試結果並揭露此限制。"
        "影像跨 split 沒有找到完全相同的像素／檔案，這不能排除近似資料；OrganAMNIST 無病患 ID 可供另行重建群組。\n\n"
        "## Quiz 2：RNN 與 LSTM 情感分析\n\n"
        "兩模型使用同一詞彙表、Embedding 128、hidden 256、dropout 0.3；"
        "最後一個有效 token 的狀態接分類層，避免 padding 影響。"
        f"AdamW、learning rate {config['text']['lr']}、batch {config['text']['batch_size']}，"
        "交叉熵與梯度裁切，依 validation Accuracy 選最佳權重，使用完整 test 推論。\n\n" +
        table(text_rows) +
        "\n\nLSTM Accuracy 較 RNN 高 7.76 個百分點，但最佳模型 train 99.46%、validation 84.06%，"
        "有明顯過擬合；錯誤預測平均信心約 93.61%，需留意過度自信。"
        "高 Accuracy 不代表機率已校準。原始指標、逐筆預測與正誤文字案例均保留。\n\n"
        "![LSTM 學習曲線](quiz2/Output/lstm/learning_curves.png)\n\n"
        "![RNN 混淆矩陣](quiz2/Output/rnn/confusion_matrix.png)\n\n"
        "![LSTM 混淆矩陣](quiz2/Output/lstm/confusion_matrix.png)\n\n"
        "詳見 [Quiz 2 報告](quiz2/Output/report.md)。\n\n"
        "## Quiz 3：ViT 與 ResNet 微調\n\n"
        "ViT-B/16 與 ResNet-50 使用 ImageNet-1K V1 預訓練權重，分類頭改為十一類；"
        "先訓練 head 再微調全模型，兩者共用相同資料分割與確定性前處理，不使用隨機擴增。"
        f"AdamW、learning rate {config['vision']['lr']}、batch {config['vision']['batch_size']}，"
        "AMP FP16、TF32，依 validation Accuracy 選權重。\n\n" + table(vision_rows) +
        "\n\nViT Accuracy 高約 0.84 個百分點、少錯 150 張；ResNet 參數較少且整批測試較快。"
        "Macro-AUC 差約 0.00030，單一 seed 不能推論普遍或統計顯著優勢。"
        "ViT kidney-left recall 約 88.23%，主要混淆之一為 kidney-left → spleen，需搭配各類指標討論。\n\n"
        "![ViT ROC](quiz3/Output/vit_b_16/roc.png)\n\n"
        "![ResNet ROC](quiz3/Output/resnet50/roc.png)\n\n"
        "![ViT 預測案例](quiz3/Output/vit_b_16/prediction_examples.png)\n\n"
        "詳見 [Quiz 3 報告](quiz3/Output/report.md)。\n\n"
        "## Quiz 4：影像描述、BLEU、時間與 Gemini\n\n"
        "共用凍結的 ImageNet ResNet-50 空間特徵編碼器，分別從頭訓練 LSTM 與 Transformer decoder。"
        "training 使用 teacher forcing，忽略 PAD 的交叉熵，validation token loss 選模型；"
        "推論只輸入圖片特徵與 BOS，不提供參考句。兩模型均測試 greedy 與 beam search（beam width=3）。\n\n" +
        table(caption) +
        "\n\nBLEU 使用全部 1,000 張測試影像與每張五段參考句、無 smoothing；上表乘以 100 顯示，"
        "不能解讀為圖片描述正確率。Beam Search 提高分數也增加解碼時間。"
        "Transformer/beam 比 LSTM/beam 高約 0.41 BLEU 點，平均 decoder 延遲約 1.65 倍。\n\n"
        "時間比較使用相同三十張圖片、batch=1、暖機、重複三次與 CUDA 同步。"
        "estimated_pipeline_ms 是分開量測的影像讀取／前處理／編碼平均時間加上解碼平均時間，"
        "不是逐張端到端計時。LSTM 訓練 30 epochs、Transformer 14 epochs，總時間不能直接當作等 epoch 效率比較。\n\n"
        "![生成描述範例](quiz4/Output/transformer/beam/caption_examples.png)\n\n"
        "### Gemini 手動審核\n\n"
        "依主體、動作、場景、數量、幻覺與重複用詞等類型人工挑選十張測試圖。"
        "每圖兩段描述匿名為 A/B、位置平衡隨機；每批五張，兩批完整回覆已匯入，沒有漏評或重複。"
        "兩段獨立評分，可同時符合或不符合；沒有向 Gemini 提供模型名稱或人工參考句。\n\n"
        "介面模型／模式：" + cases["session"]["evaluator_model_label"] +
        "（依使用者記錄，非 API 回傳的後端版本）；日期：" + cases["session"]["review_started_on"] +
        "。分數 0～5，4～5 為 match、2～3 為 partial、0～1 為 mismatch。\n\n" +
        table(cases["comparison"]) + "\n\n" + table(examples) +
        "\n\n![十張案例總覽](quiz4/Output/gemini_cases10/cases_preview.jpg)\n\n"
        "這是刻意涵蓋成功與錯誤案例的展示，40%／50% 符合率只描述本批十張圖片，"
        "不能外推整體測試集。Gemini 判斷也可能有誤；原始回覆與理由保留供人工查核。\n\n"
        "完整資料：[評估報告](quiz4/Output/gemini_cases10/report.md)、"
        "[第一批原始回覆](quiz4/Output/gemini_cases10/responses/batch_01.txt)、"
        "[第二批原始回覆](quiz4/Output/gemini_cases10/responses/batch_02.txt)、"
        "[A/B 對照](quiz4/Output/gemini_cases10/manifest.json)、"
        "[完整評分](quiz4/Output/gemini_cases10/judgments.json)。\n\n"
        "## 執行與繳交檔案\n\n"
        "Python 3.12、PyTorch 2.9.1+cu128、torchvision 0.24.1+cu128；RTX 5090 約 32 GiB 顯存、"
        "Core Ultra 7 265K、32 GB RAM。設定、環境 YAML、執行腳本與四份 Notebook 隨包提供。\n\n"
        "主要繳交包包含原始碼、資料來源與分割資訊、圖表、完整預測、評估指標、訓練曲線及 Gemini 原始回覆。"
        "六個最佳模型 best.pt 另放模型權重包，解壓後與主包相同 week4_homework/ 路徑合併。"
        "原始大型資料集、caption 特徵快取、last.pt optimizer 續訓狀態與重複 head warmup 權重未打包。"
        "重新執行須先依 README 的 prepare 階段下載並重建資料，不能把這份精簡資料包視為完整離線訓練資料。\n\n"
        "要求對照見 [RUBRIC.md](RUBRIC.md)，核對與限制詳見 [結果審查](analysis/results_review.md)。\n"
    )


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def archive(folder, target):
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    with ZipFile(target, "w", ZIP_DEFLATED, compresslevel=1) as z:
        for path in files:
            z.write(path, path.relative_to(folder.parent))
    with ZipFile(target) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"Archive CRC failure: {bad}")
    return len(files)


def main():
    if not read_json(ROOT / "completion.json").get("all_requirements_complete"):
        raise RuntimeError("Complete assignment checks before packaging")
    summary = read_json(ROOT / "quiz4/Output/gemini_cases10/summary.json")
    if not summary.get("complete") or summary.get("successful") != 20:
        raise RuntimeError("Ten-case manual review must contain all twenty real judgments")
    (ROOT / "SUBMISSION_REPORT.md").write_text(submission_report(), encoding="utf-8")
    base = ROOT / "submission"
    base.mkdir(exist_ok=True)
    number, package_name = 1, "week4_homework_submission"
    while (base / package_name).exists() or (base / (package_name + ".zip")).exists():
        number += 1
        package_name = f"week4_homework_submission_{number}"
    target = base / package_name / "week4_homework"
    target.mkdir(parents=True)
    core = ["README.md", "REPORT.md", "SUBMISSION_REPORT.md", "RUBRIC.md", "VALIDATION.md",
            "completion.json", "config.rtx5090.json", "environment.yml", "requirements.txt", "hardware.json"]
    sources = [ROOT / name for name in core] + list(ROOT.glob("*.py")) + list(ROOT.glob("*.sh")) + list(ROOT.glob("*.ps1"))
    for folder in ("common", "tests", "analysis"):
        sources.extend(p for p in (ROOT / folder).rglob("*") if p.is_file()
                       and p.suffix in (".py", ".json", ".md", ".jpg", ".png") and "__pycache__" not in p.parts)
    for quiz in ("quiz1", "quiz2", "quiz3", "quiz4"):
        for folder in ("AI", "Code", "Output"):
            for path in (ROOT / quiz / folder).rglob("*"):
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                relative = path.relative_to(ROOT / quiz / folder)
                if folder == "Output" and quiz == "quiz4" and relative.parts[0] in ("gemini", "gemini_manual"):
                    continue
                if "upload_batches" in relative.parts or path.suffix in (".pt", ".zip", ".pyc"):
                    continue
                if path.suffix in (".py", ".ipynb", ".md", ".json", ".csv", ".npz", ".png", ".jpg", ".txt", ".pdf"):
                    sources.append(path)
        data_readme = ROOT / quiz / "Data" / "README.md"
        if data_readme.exists():
            sources.append(data_readme)
    sources.extend((ROOT / "logs").glob("*.log"))
    for source in sorted(set(sources)):
        copy_file(source, target / source.relative_to(ROOT))
    # Reproducibility metadata is retained without making partial raw-data caches look complete.
    for dataset in ("imdb", "organamnist", "flickr8k"):
        copy_file(ROOT / f"quiz1/Data/{dataset}/manifest.json", target / f"quiz1/Output/data_manifests/{dataset}.json")
        vocabulary = ROOT / f"quiz1/Data/{dataset}/vocab.json"
        if vocabulary.exists():
            copy_file(vocabulary, target / f"quiz1/Data/{dataset}/vocab.json")
    flickr = ROOT / "quiz1/Data/flickr8k"
    for name in ("train.json", "val.json", "test.json", "Flickr_8k.trainImages.txt", "Flickr_8k.devImages.txt", "Flickr_8k.testImages.txt"):
        copy_file(flickr / name, target / "quiz1/Data/flickr8k" / name)
    # Keep the ten reviewed originals at their source paths as well, allowing offline revalidation.
    manifest = read_json(ROOT / "quiz4/Output/gemini_cases10/manifest.json")
    for batch in manifest["batches"]:
        for entry in batch["entries"]:
            copy_file(flickr / entry["image"], target / "quiz1/Data/flickr8k" / entry["image"])
    import torch
    split_ids = {split: torch.load(ROOT / f"quiz1/Data/imdb/{split}.pt", map_location="cpu", weights_only=True)["ids"]
                 for split in ("train", "val", "test")}
    write_json(target / "quiz1/Data/imdb/split_ids.json", split_ids)
    feature_meta = ROOT / "quiz4/Data/features/complete.json"
    copy_file(feature_meta, target / "quiz4/Output/feature_extraction_metadata.json")
    (target / "quiz1/Data/PACKAGE_NOTE.md").write_text(
        "# 資料包範圍\n\n包含詞彙表、分割 ID 與 Flickr8k 五段描述；資料 manifest 保存於 ../Output/data_manifests/。"
        "完整原始 IMDb、OrganAMNIST、Flickr8k 影像及下載壓縮檔未打包。"
        "請先執行 prepare 階段下載並重建；本包不是完整離線訓練資料。"
        "Gemini 案例的十張原圖位於 ../../quiz4/Output/gemini_cases10/uploads/。\n", encoding="utf-8")
    (target / "START_HERE.md").write_text(
        "# 繳交檔案導覽\n\n先閱讀 [作業繳交報告](SUBMISSION_REPORT.md)，"
        "再依 [要求對照](RUBRIC.md) 查閱各題 Code/ 與 Output/。\n\n"
        "- Quiz 1：資料來源、切分、類別與影像／文字樣本。\n"
        "- Quiz 2：RNN/LSTM 原始碼、曲線、測試預測、Accuracy/F1/ROC-AUC 比較。\n"
        "- Quiz 3：ViT/ResNet 微調、測試預測、各類 ROC 與 Macro-AUC。\n"
        "- Quiz 4：兩種 caption decoder、greedy/beam、BLEU、時間及十張 Gemini 手動案例。\n\n"
        "[Gemini 案例報告](quiz4/Output/gemini_cases10/report.md) 附原始圖片、提示詞與回覆，"
        "介面模型標記由使用者紀錄；本批為人工挑選案例，不是隨機抽樣效能估計。\n\n"
        "模型權重另附 models ZIP，六個 best.pt 可依相同路徑合併本包。"
        "大型原始資料、特徵快取與 optimizer 續訓檔未包含，重新執行須先下載／重建資料。\n\n"
        "完整性：FILE_MANIFEST.json 列出包內檔案大小與 SHA-256（清單自身除外）。\n", encoding="utf-8")
    payload = [{"path": p.relative_to(target).as_posix(), "bytes": p.stat().st_size, "sha256": file_hash(p)}
               for p in sorted(target.rglob("*")) if p.is_file()]
    write_json(target / "FILE_MANIFEST.json", payload)
    main_zip = base / (package_name + ".zip")
    print(f"Creating main archive ({len(payload)} files)...", flush=True)
    main_count = archive(target, main_zip)
    model_target = base / (package_name + "_models") / "week4_homework"
    model_target.mkdir(parents=True)
    model_files = []
    for quiz, kind in MODELS:
        source = ROOT / quiz / "Output" / kind / "best.pt"
        destination = model_target / source.relative_to(ROOT)
        copy_file(source, destination)
        model_files.append({"path": source.relative_to(ROOT).as_posix(), "bytes": source.stat().st_size, "sha256": file_hash(destination)})
    write_json(model_target / "MODEL_MANIFEST.json", model_files)
    (model_target / "MODELS_README.md").write_text(
        "# 最佳模型權重\n\n六個正式模型的 best.pt；維持原始檔案，不包含 last.pt、optimizer 或 head warmup 的重複權重。"
        "解壓後將 week4_homework/ 與主要繳交包的同名資料夾合併。"
        "推論還需要主包原始碼、對應詞彙表與圖片／文字；完整測試或重訓需先重建資料。"
        "SHA-256 見 MODEL_MANIFEST.json。\n", encoding="utf-8")
    model_zip = base / (package_name + "_models.zip")
    print("Creating optional archive (six best models)...", flush=True)
    archive(model_target, model_zip)
    result = {"main_folder": str(target), "main_zip": str(main_zip), "main_files": main_count,
              "main_zip_bytes": main_zip.stat().st_size, "model_zip": str(model_zip),
              "model_zip_bytes": model_zip.stat().st_size, "models": len(model_files), "archive_crc_checked": True}
    write_json(base / (package_name + "_package.json"), result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
