"""Generate safe, portable notebooks and per-quiz documentation."""
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parent
TITLES = ["資料集準備", "Sentiment Analysis", "Vision Transformer", "Image Captioning"]
STAGES = ["prepare", "sentiment", "vision", "caption"]


def main():
    for n, (title, stage) in enumerate(zip(TITLES, STAGES), 1):
        quiz = ROOT / f"quiz{n}"
        for sub in ("AI", "Code", "Data", "Output"):
            (quiz / sub).mkdir(parents=True, exist_ok=True)
        (quiz / "AI" / "AI_usage.md").write_text(
            f"# Quiz {n} AI 使用紀錄\n\n"
            "工具：OpenAI Codex。\n\n"
            "協助內容：依使用者確認之作業要求設計 PyTorch 資料處理、模型、"
            "訓練／續訓、推論、指標、圖表、報告與執行腳本。\n\n"
            "使用者確認：已確認三資料集與六模型的實作規劃，要求參考 GitHub MMIP 格式。\n\n"
            "人工確認待辦：實際執行後閱讀結果，抽查資料分割、錯誤案例、生成描述、"
            "Gemini 理由及報告結論；不將 AI 撰寫視為人工已驗證。\n\n"
            "參考：https://github.com/wuxuchengee14/MMIP\n"
            "資料與 API 來源詳見根目錄 README.md 與 quiz1/Code/prepare.py。\n",
            encoding="utf-8")
        data_text = (
            "本題原始資料自動下載至 downloads/、imdb/、organamnist/、flickr8k/。"
            "manifest.json 與 split 清單記錄來源及分割。\n"
            if n == 1 else
            "共用原始資料位於 ../../quiz1/Data/，以 Python 檔案位置解析，不依賴目前工作目錄。\n"
        )
        if n == 4:
            data_text += "features/ 是凍結 ResNet-50 的快取，附檔案 hash、影像 ID 順序與設定識別。\n"
        (quiz / "Data" / "README.md").write_text(f"# Quiz {n} Data\n\n" + data_text, encoding="utf-8")
        (quiz / "Output" / "README.md").write_text(
            f"# Quiz {n} Output\n\n執行完成後自動產生 report.md、指標、圖表與預測；"
            "模型任務另保存 best.pt、last.pt、history.csv。\n"
            "完整成果依實際執行產生，尚未訓練時不提供虛構成績。\n",
            encoding="utf-8")
        report_path = quiz / "Output" / "report.md"
        if not report_path.exists():
            report_path.write_text(f"# Quiz {n}：{title}\n\n**尚未執行。**\n\n"
                                   "執行本週流程後，本檔會以實際資料與結果更新。\n", encoding="utf-8")
        (quiz / "README.md").write_text(
            f"# Quiz {n}：{title}\n\n"
            "- AI/AI_usage.md：AI 協助與人工確認待辦。\n"
            "- Code/：本題程式與 Notebook。\n"
            "- Data/README.md：資料來源位置。\n"
            "- Output/report.md：實際執行後的報告。\n\n"
            f"從專案根目錄執行：python week4_homework/run_all.py --stage {stage}\n",
            encoding="utf-8")
        nb = nbformat.v4.new_notebook()
        nb.metadata["kernelspec"] = {"display_name": "Python (MMIP)", "language": "python", "name": "python3"}
        nb.cells = [
            nbformat.v4.new_markdown_cell(
                f"# Quiz {n}：{title}\n\n完整流程見本週 README。"
                "本 Notebook 預設只讀取成果，避免開啟時意外啟動長時間訓練或 API 呼叫。"),
            nbformat.v4.new_code_cell(
                "from pathlib import Path\nimport sys, subprocess, json\n"
                "from IPython.display import display, Markdown, Image\n"
                "candidates = [Path.cwd(), *Path.cwd().parents, Path.cwd() / 'week4_homework']\n"
                "ROOT = next(p for p in candidates if (p / 'config.rtx5090.json').is_file() "
                "and (p / 'quiz4').is_dir())\n"
                f"OUTPUT = ROOT / 'quiz{n}' / 'Output'\nprint(ROOT)"),
            nbformat.v4.new_code_cell(
                f"RUN_STAGE = False\nif RUN_STAGE:\n"
                f"    subprocess.run([sys.executable, str(ROOT / 'run_all.py'), '--stage', '{stage}'], check=True)"),
            nbformat.v4.new_code_cell(
                "report_path = OUTPUT / 'report.md'\n"
                "display(Markdown(report_path.read_text(encoding='utf-8')))"),
            nbformat.v4.new_code_cell(
                "for p in sorted(OUTPUT.rglob('*.png')):\n"
                "    print(p.relative_to(OUTPUT))\n    display(Image(filename=str(p)))"),
            nbformat.v4.new_code_cell(
                "for p in sorted(OUTPUT.rglob('metrics.json')):\n"
                "    print(p.relative_to(OUTPUT))\n    display(json.loads(p.read_text(encoding='utf-8')))"),
        ]
        if n == 4:
            nb.cells.append(nbformat.v4.new_code_cell(
                "# Set GEMINI_API_KEY in the terminal before starting Jupyter.\n"
                "RUN_GEMINI = False\nif RUN_GEMINI:\n"
                "    subprocess.run([sys.executable, str(ROOT / 'run_all.py'), '--stage', 'gemini'], check=True)"))
        nbformat.validate(nb)
        nbformat.write(nb, quiz / "Code" / f"quiz{n}.ipynb")


if __name__ == "__main__":
    main()
