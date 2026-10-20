from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parent


def main():
    stages={1:"retrieval",2:"multimodal",3:"caption"}
    for quiz,stage in stages.items():
        setup='''from pathlib import Path
import sys
from IPython.display import Markdown, Image, display
import re
ROOT = next(candidate for p in [Path.cwd(), *Path.cwd().parents] for candidate in [p, p / "week5"] if (candidate / "config.rtx5090.json").exists() and (candidate / "quiz3/Code/compare.py").is_file())
sys.path.insert(0, str(ROOT))
RUN_STAGE = False
'''
        notebook=nbformat.v4.new_notebook(cells=[
            nbformat.v4.new_markdown_cell(f"# Week5 Quiz {quiz}\n\n先閱讀根目錄 README.md。預設只讀報告，不自動下載或訓練。"),
            nbformat.v4.new_code_cell(setup),
            nbformat.v4.new_code_cell(f'''path = ROOT / "quiz{quiz}/Output/report.md"
body = path.read_text(encoding="utf-8") if path.exists() else "尚未產生報告，請先執行 run_all.py --stage report。"
display(Markdown(re.sub(r"!\\[[^\\]]*\\]\\([^)]*\\)", "", body)))
# Resolve figures from Output explicitly, so Code/ notebook working directories also work.
output = ROOT / "quiz{quiz}/Output"
for pattern in ("cub_examples.png", "comparison.png", "*/retrieval_examples.png", "*/prediction_examples.png", "*/*/caption_examples.png"):
    for figure in sorted(output.glob(pattern)):
        display(Markdown("**" + str(figure.relative_to(output)) + "**"))
        display(Image(filename=str(figure), width=1100))
'''),
            nbformat.v4.new_code_cell(f'''if RUN_STAGE:
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "run_all.py"), "--stage", "{stage}"], check=True)
'''),
            nbformat.v4.new_markdown_cell("Quiz3 的 compare 階段另行統一重算 Week4/Week5 BLEU 與 BERTScore。所有成績以正式測試輸出為準。")])
        notebook.metadata={"kernelspec":{"display_name":"Python (MMIP)","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.12"}}
        nbformat.write(notebook,ROOT/f"quiz{quiz}/Code/quiz{quiz}.ipynb")


if __name__=="__main__":main()
