"""Read-only audit of submitted Week 6 evidence; no model weights, GPU, API, or file writes."""
import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
    return value


def quiz1():
    folder=ROOT/'quiz1';runs_dir=folder/'Output/full';judge=folder/'Output/Gemini_manual'
    module('gpu_memory',folder/'Code/gpu_memory.py')
    code=module('week6_quiz1_tasks',folder/'Code/quiz1_tasks.py')
    args=SimpleNamespace(questions=folder/'Data/benchmark.json',runs_dir=runs_dir,judge_dir=judge)
    questions,runs,identity=code.load_runs(args)
    commercial=read(runs_dir/'gemini_scores.json');manifest=read(judge/'manifest.json')
    require(commercial['identity']==manifest['identity']==identity,'Quiz1: changed experiment files')
    expected={group:{} for group in runs}
    for entry in manifest['entries']:
        path=judge/entry['response'].replace('\\','/')
        require(sha(path)==commercial['response_sha256'][entry['question_id']],'Quiz1: changed review response')
        response=code.strict_json(path.read_text(encoding='utf-8-sig'))
        require(response['question_id']==entry['question_id'],'Quiz1: wrong review question')
        for rating in response['ratings']:
            expected[entry['answer_map'][rating['answer_id']]][entry['question_id']]=rating['score']
    table=read(runs_dir/'comparison.json')['rows']
    for group,run in runs.items():
        memory=code.validate_memory(run,runs_dir/(group+'.memory.json'))
        require(len(expected[group])==20,'Quiz1: incomplete commercial ratings')
        require(math.isclose(100*sum(expected[group].values())/20,commercial['groups'][group]['score_100']),'Quiz1: commercial mean')
        scores=[code.objective(q,r['answer']) for q,r in zip(questions,run['results'])]
        scores=[s for s in scores if s is not None]
        row=next(r for r in table if r['group']==group)
        require(len(scores)==10 and math.isclose(100*sum(scores)/10,row['objective_10_score']),'Quiz1: objective mean')
        require(math.isclose(memory['device_used_bytes_mean']/2**30,row['device_mean_GiB']),'Quiz1: memory mean')
    require(read(runs_dir/'completion.json')['all_requirements_complete'] is True,'Quiz1: incomplete checklist')
    return {'covered_points':30,'model_answers':80,'commercial_ratings':80,'memory_windows':80}


def quiz2():
    folder=ROOT/'quiz2';runs_dir=folder/'Output/full';judge=folder/'Output/Commercial_manual'
    module('common',folder/'Code/common.py')
    code=module('week6_quiz2_tasks',folder/'Code/quiz2_tasks.py')
    questions,runs=code.validated(folder/'Data/benchmark.json',runs_dir)
    quantitative=code.quantitative(questions,runs)
    require(quantitative==read(runs_dir/'objective_scores.json'),'Quiz2: objective scores changed')
    commercial=read(runs_dir/'commercial_scores.json');session=read(judge/'session.json')
    require(commercial['run_sha256']==session['run_sha256']==code.hashes(runs_dir),'Quiz2: stale reviews')
    for i,q in enumerate(questions):
        case=session['cases'][q['id']];case_dir=judge/'cases'/q['id']
        require(sha(case_dir/'image.png')==case['image_sha256']==q['image_sha256'],'Quiz2: image mismatch')
        require(sha(case_dir/'prompt.txt')==case['prompt_sha256'],'Quiz2: review prompt mismatch')
        path=judge/case['response'];response=code.clean_json(path.read_text(encoding='utf-8-sig'))
        require(response['question_id']==q['id'] and response['image_seen'] is True,'Quiz2: incomplete image review')
        require(sha(path)==commercial['reviews'][i]['response_sha256'],'Quiz2: review changed')
        for label,key in case['mapping'].items():
            require(response['ratings'][label]==commercial['reviews'][i]['ratings'][key],'Quiz2: anonymous mapping mismatch')
    require(read(runs_dir/'completion.json')['all_complete'] is True,'Quiz2: incomplete checklist')
    return {'covered_points':20,'model_answers':40,'commercial_ratings':40,'unique_images':18}


def quiz3():
    folder=ROOT/'quiz3'
    common=module('common',folder/'Code/common.py');common.validate_data()
    code=module('week6_quiz3_scores',folder/'Code/score_report.py')
    metrics=read(folder/'Output/metrics.json');per_question=read(folder/'Output/per_question_scores.json')
    completion=read(folder/'Output/completion.json')
    require(completion['metrics_sha256']==sha(folder/'Output/metrics.json'),'Quiz3: changed metrics')
    for kind in ['llm','vlm']:
        output=folder/'Output'/kind;training=read(output/'training.json')
        require(training['complete'] is True and not training['smoke'],'Quiz3: incomplete formal training')
        require(training['train_examples']>=100 and training['test_examples_used_for_training']==0,'Quiz3: training split')
        require(training['global_step']>0 and training['adapter_parameter_change_l2']>0,'Quiz3: no parameter updates')
        require(training['data_manifest_sha256']==sha(folder/'Data/manifest.json'),'Quiz3: data changed')
        require(sha(output/'training.json')==metrics['run_hashes'][kind]['training.json'],'Quiz3: training evidence changed')
        target=read(folder/f'Data/{kind}_test.json')
        if kind=='llm':
            questions=[dict(q,suite='target',variant='text') for q in target]+[dict(q,suite='general',variant='text') for q in read(folder/'Data/general_100.json')]
        else:questions=[dict(q,suite='visual',variant=v) for v in common.VARIANTS for q in target]
        stages={}
        for stage in ['before','after']:
            run=read(output/(stage+'.json'));stages[stage]=run
            require(sha(output/(stage+'.json'))==metrics['run_hashes'][kind][stage+'.json'],'Quiz3: changed model answers')
            require(run['complete'] is True and not run['smoke'] and len(run['results'])==len(questions),'Quiz3: incomplete test')
            require(run['training_sha256']==(sha(output/'training.json') if stage=='after' else None),'Quiz3: wrong adapter evidence')
            for q,row in zip(questions,run['results']):
                require(all(q[k]==row[k] for k in ['id','suite','variant','prompt']),'Quiz3: different task')
                method=code.llm_score if q['suite']=='target' else code.general_score if q['suite']=='general' else code.vlm_score
                score=method(q,row['answer']);group=q['suite'] if kind=='llm' else q['variant']
                stored=next(x for x in per_question if x['kind']==kind and x['stage']==stage and x['group']==group and x['id']==q['id'])
                require(all(stored[k]==v for k,v in score.items()),'Quiz3: per-question score mismatch')
                if kind=='vlm':require(row['image_sha256']==sha(folder/f"Data/variants/{q['variant']}/{q['id']}.png"),'Quiz3: variant changed')
            for group,summary in metrics['results'][kind][stage].items():
                selected=[x for x in per_question if x['kind']==kind and x['stage']==stage and x['group']==group]
                require(len(selected)==summary['questions'],'Quiz3: score denominator')
                require(math.isclose(100*sum(x['correct'] for x in selected)/len(selected),summary['accuracy_percent']),'Quiz3: accuracy mean')
        for key in ['request_digest','model','generation_config','gpu','packages','processor_config','native_token_ids','thinking']:
            require(stages['before'][key]==stages['after'][key],'Quiz3: before/after settings differ')
    require(completion['all_complete'] is True and completion['covered_points']==40,'Quiz3: checklist')
    return {'covered_points':40,'training_pairs_per_model':200,'llm_answers':400,'vlm_answers':500,
            'note':'Adapter files are intentionally excluded from GitHub; metadata and predictions verified, not omitted weight bytes.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--quiz',choices=['1','2','3','all'],default='all')
    args=parser.parse_args()
    for name,fn in [('1',quiz1),('2',quiz2),('3',quiz3)]:
        if args.quiz in [name,'all']:print('PASS Quiz '+name+': '+json.dumps(fn(),ensure_ascii=False))
    print('Read-only verification complete; no model/API calls or file changes.')


if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,FileNotFoundError) as error:raise SystemExit('FAILED: '+str(error))
