"""Sequential workflow. Full completion requires actual Gemini judgments."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from common.utils import ROOT, configure, read_json, table, write_json


def completion(config, skipped=False):
    checks = []
    for quiz, label, files in [
        ("quiz1", "Three prepared datasets", ["datasets.json", "report.md"]),
        ("quiz2", "RNN/LSTM train, inference and comparison",
         ["comparison.csv", "report.md", *[f"{k}/{p}" for k in ("rnn", "lstm") for p in
                                          ("best.pt", "complete.json", "metrics.json", "predictions.csv", "learning_curves.png")]]),
        ("quiz3", "ViT/ResNet finetune, inference and Macro-AUC",
         ["comparison.csv", "report.md", *[f"{k}/{p}" for k in ("vit_b_16", "resnet50") for p in
                                          ("best.pt", "complete.json", "metrics.json", "predictions.csv", "roc.png")]]),
        ("quiz4", "Two trained caption models, BLEU, inference timing",
         ["comparison.csv", "report.md", "encoder_timing.json", *[f"{k}/{p}" for k in ("lstm", "transformer")
          for p in ("best.pt", "complete.json", "greedy/bleu.json", "beam/bleu.json",
                    "greedy/predictions.json", "beam/predictions.json", "greedy/timing.json", "beam/timing.json")]]),
    ]:
        missing = [p for p in files if not (ROOT / quiz / "Output" / p).is_file()]
        checks.append({"requirement": label, "complete": not missing, "missing": missing})
    gemini_file = ROOT / "quiz4" / "Output" / "gemini" / "summary.json"
    gemini = read_json(gemini_file) if gemini_file.exists() else {}
    manual_file = ROOT / "quiz4" / "Output" / "gemini_manual" / "summary.json"
    manual = read_json(manual_file) if manual_file.exists() else {}
    def complete_manual(summary, expected, source):
        comparison = summary.get("comparison", [])
        return (summary.get("complete", False) and summary.get("source") == source
                and summary.get("successful") == summary.get("expected") == expected
                and len(comparison) == 2 and {r["model"] for r in comparison} == {"lstm", "transformer"}
                and all(r["successful"] == r["expected"] == expected // 2 for r in comparison))
    manual_done = complete_manual(manual, 2 * config["caption"]["gemini_samples"], "manual_upload_gemini_web")
    case_file = ROOT / "quiz4" / "Output" / "gemini_cases10" / "summary.json"
    cases = read_json(case_file) if case_file.exists() else {}
    # The user chose a 10-image case presentation in place of the 100-image API study.
    cases_done = (complete_manual(cases, 20, "manual_upload_gemini_web_case_study")
                  and cases.get("selection_method") == "purposive_case_study")
    gemini_done = (gemini.get("complete", False) or manual_done or cases_done) and not skipped
    checks.append({"requirement": "Actual Gemini image-caption judgments",
                   "complete": gemini_done,
                   "source": ("manual_upload_gemini_web_case_study" if cases_done else
                              "manual_upload_gemini_web" if manual_done else "Gemini API"),
                   "scope": "10 selected images / 20 judgments; case presentation only" if cases_done else "100-image study",
                   "missing": [] if gemini_done else ["Gemini evaluation pending"]})
    done = all(c["complete"] for c in checks)
    write_json(ROOT / "completion.json", {"all_requirements_complete": done, "checks": checks, "config": config})
    (ROOT / "REPORT.md").write_text(
        "# Week 4 執行結果\n\n" +
        ("全部要求已產生實際成果。" if done else "**尚有項目未完成，請依 completion.json 補齊。**") +
        "\n\n" + table([{"requirement": c["requirement"], "complete": c["complete"],
                         "missing": ", ".join(c["missing"])} for c in checks]) +
        "\n\n- [Quiz 1](quiz1/Output/report.md)\n- [Quiz 2](quiz2/Output/report.md)\n"
        "- [Quiz 3](quiz3/Output/report.md)\n- [Quiz 4](quiz4/Output/report.md)\n",
        encoding="utf-8")
    if cases_done:
        with (ROOT / "REPORT.md").open("a", encoding="utf-8") as f:
            f.write("\n依使用者指定，Gemini 要求採 10 張人工挑選案例、兩模型共 20 筆網頁手動評估完成。"
                    "這是案例展示，非原 100 張抽樣研究或 API 全部完成；符合率不能外推全體測試集。"
                    "詳見 [10 張案例報告](quiz4/Output/gemini_cases10/report.md)。\n")
    elif manual_done:
        with (ROOT / "REPORT.md").open("a", encoding="utf-8") as f:
            f.write("\nGemini 要求以手動上傳網頁評估完成，非 API 全部完成；"
                    "詳見 [手動評估報告](quiz4/Output/gemini_manual/report.md)。\n")
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.rtx5090.json")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--stage", choices=("all", "prepare", "sentiment", "vision", "caption", "gemini", "report"),
                        default="all")
    parser.add_argument("--skip-gemini", action="store_true", help="Run local work; full-completion exit code remains 3.")
    args = parser.parse_args()
    config = read_json(args.config)
    device = configure(config, args.device)
    stage = args.stage
    if stage == "all" and not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is unset: local tasks will run; Gemini remains pending (exit 3).", flush=True)
    tick = time.perf_counter()
    if stage in ("all", "prepare"):
        from quiz1.Code.prepare import run
        print("\n[1/5] Prepare datasets", flush=True)
        run(config)
    if stage in ("all", "sentiment"):
        from quiz2.Code.train import run
        print("\n[2/5] Train RNN and LSTM", flush=True)
        run(config, device)
    if stage in ("all", "vision"):
        from quiz3.Code.train import run
        print("\n[3/5] Finetune ViT and ResNet", flush=True)
        run(config, device)
    if stage in ("all", "caption"):
        from quiz4.Code.train import run
        print("\n[4/5] Train two caption models, evaluate greedy/beam", flush=True)
        run(config, device)
    if stage in ("all", "gemini"):
        from quiz4.Code.gemini import run
        from quiz4.Code.train import make_report
        print("\n[5/5] Gemini image-caption evaluation", flush=True)
        run(config, skip=args.skip_gemini)
        make_report()
    done = completion(config, skipped=args.skip_gemini)
    print(f"\nStage {stage} finished in {time.perf_counter()-tick:.1f}s. Reports: {ROOT}", flush=True)
    if stage in ("all", "gemini", "report") and not done:
        print("INCOMPLETE: see completion.json. Set GEMINI_API_KEY and rerun --stage gemini if needed.", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
