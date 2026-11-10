"""Sequential GPU workflow; never overwrite evidence or count a smoke run as completion."""
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def run(script,*args):
    command=[sys.executable,'-X','utf8',str(ROOT/'Code'/script),*args]
    print('RUN: '+' '.join(command),flush=True)
    subprocess.run(command,check=True)


def complete(path):
    if not path.exists():return False
    data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('complete') is not True or data.get('smoke'):
        raise RuntimeError('Incomplete evidence exists. Inspect and preserve it before restarting: '+str(path))
    print('Keeping completed evidence: '+str(path),flush=True)
    return True


def main():
    run('prepare_data.py');run('preflight.py')
    for kind in ['llm','vlm']:
        if not complete(ROOT/f'Output/{kind}/before.json'):run('evaluate.py','--kind',kind,'--stage','before')
        if not complete(ROOT/f'Output/{kind}/training.json'):run('train_dpo.py','--kind',kind)
        if not complete(ROOT/f'Output/{kind}/after.json'):run('evaluate.py','--kind',kind,'--stage','after')
    run('score_report.py','report');run('plots.py');run('package_submission.py')
    print('Finished all six basic/advanced requirements. See Output/report.md and completion.json.',flush=True)


if __name__=='__main__':main()
