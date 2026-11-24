"""Package complete coursework evidence, excluding base weights and intermediate checkpoints."""
import zipfile
from common import ROOT,read,require,save,sha


def main():
    checklist=read(ROOT/'Output/completion.json')
    require(checklist['all_complete'] is True and checklist['covered_points']==40,'Complete and verify all six requirements first')
    require(checklist['metrics_sha256']==sha(ROOT/'Output/metrics.json'),'Scores changed after completion check')
    files=[]
    for folder in ['Code','Data']:
        files += [p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    for name in ['README.md','requirements.txt','run_quiz3.ps1','run_quiz3.sh']:
        files.append(ROOT/name)
    for name in ['metrics.json','per_question_scores.json','comparison.csv','completion.json','report.md']:
        files.append(ROOT/'Output'/name)
    files += [p for p in (ROOT/'Output/figures').rglob('*') if p.is_file()]
    for kind in ['llm','vlm']:
        files += [ROOT/f'Output/{kind}/{name}' for name in ['training.json','before.json','after.json']]
        for folder in ['adapter','processor']:
            files += [p for p in (ROOT/f'Output/{kind}/{folder}').rglob('*') if p.is_file()]
    require(all(p.is_file() for p in files),'Submission evidence missing')
    manifest={p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(set(files))}
    save(ROOT/'Output/submission_manifest.json',manifest)
    output=ROOT/'quiz3_submission.zip';temporary=output.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in sorted(set(files)):archive.write(p,'quiz3/'+p.relative_to(ROOT).as_posix())
        archive.write(ROOT/'Output/submission_manifest.json','quiz3/Output/submission_manifest.json')
    temporary.replace(output)
    print(f'Submission: {output} ({output.stat().st_size/2**20:.1f} MiB)',flush=True)


if __name__=='__main__':main()
