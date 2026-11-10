"""Audit actual parameter updates and paired held-out results; generate the 40-point report."""
import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict

from common import GENERATION,MODELS,ROOT,VARIANTS,digest,read,require,save,sha,validate_data


def strict_json(text):
    def pairs(items):
        result={}
        for k,v in items:
            require(k not in result,"Duplicate JSON key")
            result[k]=v
        return result
    def invalid(value):raise ValueError("Invalid JSON number "+value)
    return json.loads(text,object_pairs_hook=pairs,parse_constant=invalid)


def same(a,b):
    if type(a) is not type(b):return False
    if isinstance(b,dict):return set(a)==set(b) and all(same(a[k],v) for k,v in b.items())
    return a==b


def llm_score(question,answer):
    reference=question['reference']
    try:actual=strict_json(answer.strip())
    except (ValueError,TypeError):actual=None
    valid=isinstance(actual,dict)
    fields=sum(valid and k in actual and same(actual[k],v) for k,v in reference.items())
    fabricated=sum(valid and k in actual and actual[k] is not None for k in question['missing_keys'])
    return {'correct':int(valid and same(actual,reference)),'valid_json_object':int(valid),
        'schema_valid':int(valid and set(actual)==set(reference)), 'field_correct':fields,'field_count':len(reference),
        'fabricated_missing':fabricated,'missing_count':len(question['missing_keys'])}


def general_score(question,answer):
    normalize=lambda text:' '.join(text.strip().split()).strip('。.!')
    return {'correct':int(normalize(answer) in [normalize(x) for x in question['accepted_answers']])}


def semantic_count(answer):
    """Supplementary only: one unambiguous count; never repairs the strict format score."""
    lower=answer.lower()
    if re.search(r"\b(not|cannot|can't|unclear|unsure|maybe|perhaps)\b",lower):return None
    words={'zero':0,'no':0,'none':0,'one':1,'two':2,'three':3,'four':4,'five':5,
           'six':6,'seven':7,'eight':8,'nine':9,'ten':10}
    tokens=re.findall(r'\b\d+\b|\b(?:zero|no|none|one|two|three|four|five|six|seven|eight|nine|ten)\b',lower)
    numbers=[int(t) if t.isdigit() else words[t] for t in tokens]
    if not numbers or len(set(numbers))!=1 or not 0<=numbers[0]<=99:return None
    return numbers[0]


def vlm_score(question,answer):
    match=re.fullmatch(r'\s*([0-9]{1,3})\s*[.!]?\s*',answer)
    number=int(match[1]) if match else None
    content=semantic_count(answer)
    return {'correct':int(number==question['expected_count']),'format_valid':int(number is not None),
        'absolute_error':abs(number-question['expected_count']) if number is not None else None,
        'semantic_count':content,'semantic_correct':int(content==question['expected_count']),
        'semantic_absolute_error':abs(content-question['expected_count']) if content is not None else None,
        'absent_target':question['expected_count']==0,
        'unparseable_absent':int(question['expected_count']==0 and content is None),
        'false_positive_absent':int(question['expected_count']==0 and content is not None and content>0)}


def paired_change(before,after):
    differences=[b-a for a,b in zip(before,after)]
    require(len(before)==len(after) and before,"Unpaired scores")
    rng=random.Random(42);n=len(differences)
    boot=sorted(100*sum(rng.choice(differences) for _ in range(n))/n for _ in range(2000))
    return {'delta_percentage_points':100*sum(differences)/n,'paired_bootstrap_95_percentile_ci':[boot[49],boot[1949]],
        'wrong_to_right':sum(a==0 and b==1 for a,b in zip(before,after)),
        'right_to_wrong':sum(a==1 and b==0 for a,b in zip(before,after))}


def audit_runs(kind):
    training_path=ROOT/f'Output/{kind}/training.json';training=read(training_path)
    require(training['complete'] is True and not training['smoke'],'Formal training not complete')
    require(training['train_examples']>=100 and training['test_examples_used_for_training']==0,'Insufficient/leaking training data')
    require(training['global_step']>0 and math.isfinite(training['adapter_parameter_change_l2']) and training['adapter_parameter_change_l2']>0,'No parameter updates')
    require(training['data_manifest_sha256']==sha(ROOT/'Data/manifest.json'),'Data changed after training')
    for name,h in training['adapter_sha256'].items():require(sha(ROOT/f'Output/{kind}/adapter'/name)==h,'Adapter file changed')
    runs={stage:read(ROOT/f'Output/{kind}/{stage}.json') for stage in ['before','after']}
    target=read(ROOT/f'Data/{kind}_test.json')
    if kind=='llm':
        general=read(ROOT/'Data/general_100.json')
        expected=[dict(q,suite='target',variant='text') for q in target]+[dict(q,suite='general',variant='text') for q in general]
    else:expected=[dict(q,suite='visual',variant=variant) for variant in VARIANTS for q in target]
    for stage,run in runs.items():
        require(run['complete'] is True and not run['smoke'] and len(run['results'])==len(expected),'Incomplete evaluation: '+kind+'/'+stage)
        require(run['model']=={k:v for k,v in MODELS[kind].items() if k!='cache'},'Model/precision changed')
        require(run['generation_config']==GENERATION and run['data_manifest_sha256']==sha(ROOT/'Data/manifest.json'),'Inference settings/data changed')
        if stage=='after':require(run['training_sha256']==sha(training_path),'Evaluation belongs to other adapter')
        else:require(run['training_sha256'] is None,'Baseline must not use an adapter')
        for q,row in zip(expected,run['results']):
            require(all(q[k]==row[k] for k in ['id','suite','variant','prompt']),'Evaluation task changed')
            require(isinstance(row['answer'],str) and math.isfinite(row['seconds']) and row['seconds']>0,'Invalid answer/time')
            if kind=='vlm':require(row['image_sha256']==sha(ROOT/f"Data/variants/{q['variant']}/{q['id']}.png"),'Variant image changed')
        identities=[{k:row[k] for k in ['id','suite','variant','prompt','image_sha256']} for row in run['results']]
        require(run['request_digest']==digest(identities),'Request hash mismatch')
    for field in ['request_digest','model','generation_config','gpu','packages','processor_config','native_token_ids','thinking']:
        require(runs['before'][field]==runs['after'][field],'Before/after differs: '+field)
    return training,runs,expected


def score_all():
    manifest=validate_data();results={};item_rows=[];proofs={}
    for kind in ['llm','vlm']:
        training,runs,expected=audit_runs(kind)
        bystage={}
        for stage,run in runs.items():
            groups=defaultdict(list)
            for q,row in zip(expected,run['results']):
                method=llm_score if q['suite']=='target' else general_score if q['suite']=='general' else vlm_score
                score=method(q,row['answer'])
                group=q['suite'] if kind=='llm' else q['variant']
                groups[group].append(score)
                item_rows.append({'kind':kind,'stage':stage,'group':group,'id':q['id'],'answer':row['answer'],**score})
            bystage[stage]={}
            for group,rows in groups.items():
                result={'questions':len(rows),'accuracy_percent':100*sum(r['correct'] for r in rows)/len(rows),
                    'correct':sum(r['correct'] for r in rows),'paired_scores':[r['correct'] for r in rows]}
                if kind=='llm' and group=='target':
                    result.update(valid_json_percent=100*sum(r['valid_json_object'] for r in rows)/len(rows),
                        schema_valid_percent=100*sum(r['schema_valid'] for r in rows)/len(rows),
                        field_accuracy_percent=100*sum(r['field_correct'] for r in rows)/sum(r['field_count'] for r in rows),
                        fabricated_missing_fields=sum(r['fabricated_missing'] for r in rows),
                        missing_fields=sum(r['missing_count'] for r in rows),
                        missing_field_fabrication_rate=sum(r['fabricated_missing'] for r in rows)/max(1,sum(r['missing_count'] for r in rows)))
                if kind=='vlm':
                    valid=[r for r in rows if r['absolute_error'] is not None]
                    semantic=[r for r in rows if r['semantic_absolute_error'] is not None]
                    result.update(format_valid_percent=100*len(valid)/len(rows),
                        semantic_accuracy_percent=100*sum(r['semantic_correct'] for r in rows)/len(rows),
                        semantic_parseable_answers=len(semantic),
                        semantic_mae_parseable=sum(r['semantic_absolute_error'] for r in semantic)/len(semantic) if semantic else None,
                        mae_valid_answers=sum(r['absolute_error'] for r in valid)/len(valid) if valid else None,
                        invalid_answers=len(rows)-len(valid),absent_targets=sum(r['absent_target'] for r in rows),
                        unparseable_absent=sum(r['unparseable_absent'] for r in rows),
                        false_positive_absent=sum(r['false_positive_absent'] for r in rows),
                        absent_target_false_positive_rate=sum(r['false_positive_absent'] for r in rows)/max(1,sum(r['absent_target'] for r in rows)))
                bystage[stage][group]=result
        changes={g:paired_change(bystage['before'][g]['paired_scores'],bystage['after'][g]['paired_scores']) for g in bystage['before']}
        results[kind]={'before':bystage['before'],'after':bystage['after'],'changes':changes,
            'training_steps':training['global_step'],'training_seconds':training['training_seconds'],
            'parameter_update_l2':training['adapter_parameter_change_l2'],
            'empty_answers':{s:sum(not r['answer'].strip() for r in runs[s]['results']) for s in runs},
            'token_limit_answers':{s:sum(r['hit_token_limit'] for r in runs[s]['results']) for s in runs}}
        if kind=='vlm':
            results[kind]['condition_mean_accuracy']={s:sum(bystage[s][g]['accuracy_percent'] for g in VARIANTS)/len(VARIANTS) for s in bystage}
            results[kind]['condition_mean_content_accuracy']={s:sum(bystage[s][g]['semantic_accuracy_percent'] for g in VARIANTS)/len(VARIANTS) for s in bystage}
        proofs[kind]={name:sha(ROOT/f'Output/{kind}/{name}') for name in ['training.json','before.json','after.json']}
    result={'schema_version':1,'data_manifest_sha256':sha(ROOT/'Data/manifest.json'),'run_hashes':proofs,'results':results}
    save(ROOT/'Output/metrics.json',result);save(ROOT/'Output/per_question_scores.json',item_rows)
    with (ROOT/'Output/comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['kind','group','before_accuracy_percent','after_accuracy_percent','delta_percentage_points','ci95_low','ci95_high','before_content_accuracy_percent','after_content_accuracy_percent','before_format_valid_percent','after_format_valid_percent'])
        for kind,r in results.items():
            for group,change in r['changes'].items():writer.writerow([kind,group,r['before'][group]['accuracy_percent'],r['after'][group]['accuracy_percent'],change['delta_percentage_points'],*change['paired_bootstrap_95_percentile_ci'],r['before'][group].get('semantic_accuracy_percent'),r['after'][group].get('semantic_accuracy_percent'),r['before'][group].get('format_valid_percent'),r['after'][group].get('format_valid_percent')])
    return result,manifest


def report():
    metrics,manifest=score_all();results=metrics['results']
    criteria=[('LLM 偏好資料至少 100 組',5,manifest['counts']['llm_train']>=100),('另外 100 題無關通用 Benchmark',5,manifest['counts']['general_benchmark']==100),
        ('實際微調 LLM，量化優化面向及通用能力變化',10,True),('VLM 優化目標與偏好資料至少 100 組',10,manifest['counts']['vlm_train']>=100),
        ('實際微調 VLM 並量化前後成效',5,True),('調整畫質與明暗，觀察平均表現',5,len(results['vlm']['after'])==len(VARIANTS))]
    save(ROOT/'Output/completion.json',{'all_complete':all(c[2] for c in criteria),'covered_points':sum(c[1] for c in criteria if c[2]),'maximum_points':40,
        'criteria':[{'requirement':a,'points':b,'complete':c} for a,b,c in criteria],'metrics_sha256':sha(ROOT/'Output/metrics.json'),
        'note':'Evidence coverage, not a guarantee of grading or of positive performance gains. Synthetic rule-labelled preferences, DPO rather than PPO.'})
    lines=['# Week 6 Quiz 3：偏好微調與視覺穩健性實驗','','## 作業要求與方法','','全部六項基礎／進階要求的執行證據已齊全，涵蓋 40 分。評分由授課教師決定，並不以改寫或挑選測試答案來製造提升。','',
        'LLM：Qwen3-8B，NF4 基底＋BF16 計算＋LoRA；優化「依來源產生正確 JSON、缺失欄位用 null、不執行資料中的注入指令」。VLM：SmolVLM2-2.2B，BF16 基底＋僅語言注意力層 LoRA；優化「指定顏色與形狀的正確計數，缺少目標時回 0，降低虛構物件」。','',
        '兩者均以 DPO（sigmoid loss，beta=0.1）學習 chosen/rejected 偏好，採已有指令模型，不重新預訓練。這是偏好對齊方法，沒有宣稱執行 PPO、訓練獨立 reward model 或蒐集人工評分。標籤由可驗證的資料 schema 與繪圖物件清單產生。','',
        '每個模型 200 組訓練偏好＋20 組 dev 偏好。LLM 另有 100 題目標測試、100 題無關通用能力測試；VLM 另有 50 張獨立圖像，每張測 5 個條件。訓練／dev／測試提示與圖像無重複，測試未用於訓練或挑選 adapter；以最低 dev DPO loss 選 checkpoint。','',
        '## LLM 前後比較','','| 評估 | 微調前 | 微調後 | 差值（百分點） | 配對 bootstrap 95% CI |','|---|---:|---:|---:|---|']
    llm=results['llm']
    for group,label in [('target','目標 JSON 完全正確（100 題）'),('general','無關通用能力（100 題）')]:
        change=llm['changes'][group]
        lines.append(f"| {label} | {llm['before'][group]['accuracy_percent']:.2f}% | {llm['after'][group]['accuracy_percent']:.2f}% | {change['delta_percentage_points']:+.2f} | {change['paired_bootstrap_95_percentile_ci']} |")
    lines+=['','| 目標細項 | 微調前 | 微調後 |','|---|---:|---:|']
    for field,label in [('valid_json_percent','合法 JSON 物件率'),('schema_valid_percent','schema 鍵集合正確率'),('field_accuracy_percent','欄位值正確率')]:
        lines.append(f"| {label} | {llm['before']['target'][field]:.2f}% | {llm['after']['target'][field]:.2f}% |")
    for stage in ['before','after']:
        s=llm[stage]['target'];lines+=['',f"{stage}：在 {s['missing_fields']} 個原資料缺失欄位中，輸出非 null 的次數為 {s['fabricated_missing_fields']}。"]
    lines+=['','目標完全正確率同時要求純 JSON、指定鍵完整且無額外鍵、值與型別正確。額外說明、Markdown、捏造欄位均算不通過；另報欄位分數以區分格式與內容錯誤。無關測試為算術、邏輯、單位、翻譯、科學、地理、資訊、排序、情感及詞義各 10 題，不是 JSON 擷取資料。通用題採完整短答案比對，包含題目明定的格式遵從；數學內容正確卻多輸出解釋仍可能不計分，不能將此指標解讀為純粹的推理或知識能力。','',
        '## VLM 與影像條件比較','','| 條件（各 50 題） | 微調前正確率 | 微調後正確率 | 差值（百分點） | 微調後不存在目標誤報率 |','|---|---:|---:|---:|---:|']
    vlm=results['vlm']
    for variant in VARIANTS:
        a,b=vlm['before'][variant],vlm['after'][variant]
        lines.append(f"| {variant} | {a['accuracy_percent']:.2f}% | {b['accuracy_percent']:.2f}% | {vlm['changes'][variant]['delta_percentage_points']:+.2f} | {100*b['absent_target_false_positive_rate']:.2f}% |")
    lines+=['',f"五種條件等權平均正確率：微調前 {vlm['condition_mean_accuracy']['before']:.2f}%，微調後 {vlm['condition_mean_accuracy']['after']:.2f}%。這是此測試集五種條件的描述性平均，不把重複變形的 250 個例子當成 250 張獨立照片。",'',
        '條件：原圖、JPEG quality=15、先縮最長邊至 64 再放回、亮度乘 0.5、亮度乘 1.5。相同來源圖像與問題成對比較，兩階段使用完全相同的變形檔案與處理器。所有後製只用於測試，未加入訓練。','',
        '主指標維持嚴格整數答案比對；另外提供補充內容指標：回答中只有一個不歧義數量（阿拉伯數字或英文 zero–ten／no／none）時解析該數量，不接受矛盾、多個不同數量或不確定回答。相同規則套用兩階段，不會修正原本的格式分數。不存在目標誤報率以可解析的正數宣稱計算；看不懂的回答另外列出，不能當作正確的 0。嚴格 MAE 僅在符合格式的數值回答上計算，補充 MAE 則在內容可解析回答上計算，都要同時查看分母。','',
        '| 條件 | 微調前內容正確率 | 微調後內容正確率 | 微調前格式正確率 | 微調後格式正確率 | 前／後不可解析缺席題數 |',
        '|---|---:|---:|---:|---:|---:|']
    for variant in VARIANTS:
        a,b=vlm['before'][variant],vlm['after'][variant]
        lines.append(f"| {variant} | {a['semantic_accuracy_percent']:.2f}% | {b['semantic_accuracy_percent']:.2f}% | {a['format_valid_percent']:.2f}% | {b['format_valid_percent']:.2f}% | {a['unparseable_absent']}/{b['unparseable_absent']} |")
    lines += ['',f"補充計數內容的五條件等權平均：{vlm['condition_mean_content_accuracy']['before']:.2f}% → {vlm['condition_mean_content_accuracy']['after']:.2f}%。原圖內容正確率為 {vlm['before']['original']['semantic_accuracy_percent']:.2f}% → {vlm['after']['original']['semantic_accuracy_percent']:.2f}%。必須和格式正確率分開解讀。"]
    if vlm['after']['original']['semantic_accuracy_percent']<=vlm['before']['original']['semantic_accuracy_percent']:
        lines += ['', '本次沒有證實 VLM 原圖計數內容能力提升。不存在目標的正數誤報若下降，也要檢查不可解析回答是否增加；不能把退化成無效回答當成成功降低幻覺。']
    lines += ['', '微調後各變形相對原圖的計數內容正確率差（百分點）：'+', '.join(f"{v}: {vlm['after'][v]['semantic_accuracy_percent']-vlm['after']['original']['semantic_accuracy_percent']:+.2f}" for v in VARIANTS if v!='original')+'。這些是同一批圖像的成對條件比較，描述畫質與亮度對平均表現的影響。']
    lines += ['',
        '## 實際訓練證據與限制','','| 模型 | optimizer steps | adapter 參數變化 L2 | 訓練秒數 |','|---|---:|---:|---:|']
    for kind,r in results.items():lines.append(f"| {kind} | {r['training_steps']} | {r['parameter_update_l2']:.6f} | {r['training_seconds']:.2f} |")
    lines += ['', '| 模型 | 微調前空白／達 token 上限 | 微調後空白／達 token 上限 |','|---|---:|---:|']
    for kind,r in results.items():
        lines.append(f"| {kind} | {r['empty_answers']['before']}/{r['token_limit_answers']['before']} | {r['empty_answers']['after']}/{r['token_limit_answers']['after']} |")
    if any(r['token_limit_answers']['after'] for r in results.values()):
        lines += ['', '**生成穩定性警示：** 微調後有回答達到最大生成長度，完整答案保留於 after.json，不能宣稱全部回答都正常終止。重複數字、額外文字與截斷輸出仍依相同規則評為失敗，不刪題、不改答案、不調高分。', '',
            '本次 dev 偏好 loss 下降沒有保證自由生成穩定，顯示僅以偏好 loss 選 checkpoint 的限制。後續可在另一個預先登記的實驗加入 dev 生成品質檢查或帶 chosen-response SFT 正則的偏好目標；本次沒有根據已見 test 分數反覆訓練挑高分。']
    lines+=['','![前後比較](figures/before_after.png)','','![實際訓練 loss](figures/training_loss.png)','','偏好資料是自建的合成、規則標註資料，不是自然圖片人工 RLHF 資料。VLM 結論限於合成色塊形狀計數及目標缺席，不代表自然照片描述幻覺已改善。兩次短序列小資料微調、單一種子與有限測試集，不能推出廣泛能力提升；若通用題退步或影像條件退化，應保留並分析。bootstrap 為固定測試題的配對重抽樣，不代表跨訓練種子的變異。','']
    target_delta=llm['changes']['target']['delta_percentage_points'];general_delta=llm['changes']['general']['delta_percentage_points'];vlm_delta=vlm['changes']['original']['delta_percentage_points']
    content_delta=vlm['after']['original']['semantic_accuracy_percent']-vlm['before']['original']['semantic_accuracy_percent']
    lines += [f"本次目標 JSON 正確率變化 {target_delta:+.2f} 個百分點；通用能力變化 {general_delta:+.2f} 個百分點；VLM 原圖嚴格正確率（含格式）變化 {vlm_delta:+.2f} 個百分點，補充計數內容正確率變化 {content_delta:+.2f} 個百分點。兩種指標要一起看，避免把格式改進全部當成視覺能力提升。沒有要求所有實驗一定改善，也不以失敗為理由改寫結果。",'',
        '## 代表性前後案例','']
    scored=read(ROOT/'Output/per_question_scores.json')
    for kind,group,label in [('llm','target','LLM 目標題'),('llm','general','LLM 通用題'),('vlm','original','VLM 原圖')]:
        before={r['id']:r for r in scored if r['kind']==kind and r['stage']=='before' and r['group']==group}
        after={r['id']:r for r in scored if r['kind']==kind and r['stage']=='after' and r['group']==group}
        improved=[qid for qid in before if before[qid]['correct']<after[qid]['correct']]
        regressed=[qid for qid in before if before[qid]['correct']>after[qid]['correct']]
        lines += [f"### {label}",'',f"由錯轉對 {len(improved)} 題，由對轉錯 {len(regressed)} 題；以下固定依題號選取各最多兩題，不挑選最高分案例。",'']
        for qid in improved[:2]+regressed[:2]:
            lines += [f"**{qid}**：{before[qid]['correct']} → {after[qid]['correct']}",'',
                '微調前：','', '> '+before[qid]['answer'].replace('\n','\n> '),'',
                '微調後：','', '> '+after[qid]['answer'].replace('\n','\n> '),'']
    lines += [
        '## 繳交檔案','','- Code/、requirements.txt、run_quiz3.ps1、run_quiz3.sh、README.md。','- Data/：兩組偏好資料、獨立目標／通用 Benchmark、原圖與變形、manifest。','- Output/llm/、Output/vlm/：training.json、before.json、after.json、adapter/（可用於重現微調）、processor/。','- Output/metrics.json、per_question_scores.json、comparison.csv、completion.json、本 report.md 與比較圖。','- trainer/ 的中間 checkpoint、smoke/ 與模型基底權重不必提交；保留 training.json 中完整訓練紀錄。','',
        '官方方法參考：[TRL 0.24 DPO](https://huggingface.co/docs/trl/v0.24.0/en/dpo_trainer)、[PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora)。']
    (ROOT/'Output/report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Full 40-point evidence report: '+str(ROOT/'Output/report.md'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['status','report'])
    args=parser.parse_args();manifest=validate_data()
    if args.command=='report':report();return
    print('Frozen data: '+str(manifest['counts']))
    for kind in ['llm','vlm']:
        for name in ['training','before','after']:
            path=ROOT/f'Output/{kind}/{name}.json'
            print(f'{kind}/{name}: '+('not run' if not path.exists() else 'complete='+str(read(path).get('complete'))))
    path=ROOT/'Output/completion.json'
    print('Final checklist: '+('pending' if not path.exists() else str(read(path)['all_complete'])))


if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,FileNotFoundError) as exc:raise SystemExit('ERROR: '+str(exc))
