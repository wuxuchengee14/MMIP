"""Audit real saved outputs independently of the training loop; no model inference."""
from pathlib import Path
import csv
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from common.utils import ROOT,read_json,write_json


def require(condition,message):
    if not condition:raise ValueError(message)


def near(actual,expected,label):
    require(np.isclose(actual,expected,rtol=1e-6,atol=1e-7),f"{label}: {actual} != {expected}")


def main():
    checks=[]
    config=read_json(ROOT/"config.rtx5090.json")
    flickr=read_json(ROOT/'quiz3/Data/flickr_manifest.json')
    from collections import defaultdict
    original=Path(flickr['source_directory'])
    image_splits={r['id']:s for s in ('train','val','test') for r in read_json(original/f'{s}.json')}
    hashes=defaultdict(list)
    for sid,sha in flickr['image_hashes'].items():hashes[sha].append(sid)
    duplicates=[ids for ids in hashes.values() if len(ids)>1]
    cross=[ids for ids in duplicates if len({image_splits[sid] for sid in ids})>1]
    require(not cross,'Exact Flickr image duplicates across splits')
    write_json(ROOT/'quiz3/Output/data_audit.json',{'images':len(image_splits),'cross_split_exact_duplicates':len(cross),
        'within_split_duplicate_groups':[[{'id':sid,'split':image_splits[sid]} for sid in ids] for ids in duplicates]})
    records=read_json(ROOT/"quiz1/Data/cub/test.json")
    mapping={r['id']:r for r in records}
    split=read_json(ROOT/"quiz1/Data/cub/retrieval_splits.json")['test']
    require(len(records)==5794 and len(split['query'])==2878 and len(split['gallery'])==2916,"Incomplete CUB test data")
    for kind in config['retrieval']['backbones']:
        folder=ROOT/"quiz1/Output"/kind
        metrics=read_json(folder/'metrics.json')
        groups={sid:[] for sid in split['query']}
        with (folder/'retrieval_results.csv').open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream):
                q,g=row['query_id'],row['gallery_id']
                require(q!=g and q in groups and g in split['gallery'],"Invalid query/gallery ID")
                correct=mapping[q]['label']==mapping[g]['label']
                require((row['relevant']=='True')==correct,"Incorrect relevance label")
                groups[q].append((int(row['rank']),float(row['similarity']),correct))
        ap=[];hits=[];precision=[];recall=[]
        for sid,rows in groups.items():
            require([r[0] for r in rows]==list(range(1,11)),"Missing/duplicate Top-10 ranks")
            require(all(a[1]>=b[1] for a,b in zip(rows,rows[1:])),"Unsorted similarity")
            relevant=np.array([r[2] for r in rows])
            count=sum(mapping[g]['label']==mapping[sid]['label'] for g in split['gallery'])
            ap.append(float(((relevant.cumsum()/np.arange(1,11))*relevant).sum()/min(count,10)))
            hits.append(bool(relevant[0]));precision.append(relevant.mean());recall.append(relevant.sum()/count)
        for key,value in [('map_at_10',np.mean(ap)),('hit_at_1',np.mean(hits)),
                          ('precision_at_10',np.mean(precision)),('recall_at_10',np.mean(recall))]:
            near(metrics[key],value,f'{kind} {key}')
        with np.load(folder/'embeddings.npz') as cache:
            require(cache['features'].shape==(5794,config['retrieval']['embedding_dim']),"Wrong embedding shape")
            require(np.isfinite(cache['features']).all(),"Nonfinite embeddings")
            require(list(cache['ids'])==[r['id'] for r in records],"Embedding ID mismatch")
        checks.append({'task':'retrieval','model':kind,'queries':len(groups),'map_at_10':metrics['map_at_10']})
    for row in read_json(ROOT/'quiz2/Output/comparison.json'):
        kind=row['image_backbone'];folder=ROOT/'quiz2/Output'/kind
        with np.load(folder/'probabilities.npz') as cache:
            probs,labels=cache['probabilities'],cache['labels']
            require(probs.shape==(5794,200) and np.isfinite(probs).all(),"Bad classification probabilities")
            require(np.allclose(probs.sum(1),1,atol=1e-5),"Probabilities do not sum to one")
            require(list(cache['ids'])==[r['id'] for r in records],"Classification IDs disagree")
            require(np.array_equal(labels,[r['label'] for r in records]),"Classification labels disagree")
            predicted=probs.argmax(1)
            near(row['accuracy'],accuracy_score(labels,predicted),f'{kind} accuracy')
            near(row['macro_f1'],f1_score(labels,predicted,average='macro'),f'{kind} macro F1')
            matrix=np.load(folder/'confusion_matrix.npy')
            require(matrix.shape==(200,200) and matrix.sum()==5794,"Incomplete confusion matrix")
            require(matrix.trace()==int((labels==predicted).sum()),"Confusion matrix trace mismatch")
        checks.append({'task':'multimodal','model':kind,'samples':5794,'accuracy':row['accuracy']})
    compared=read_json(ROOT/'quiz3/Output/comparison.json')
    require(len(compared)==6 and len({r['bertscore_hash'] for r in compared})==1,"Caption methods/evaluator mismatch")
    flickr_root=(ROOT/config['week4_root']).resolve()/'quiz1/Data/flickr8k'
    test=read_json(flickr_root/'test.json')
    from quiz3.Code.compare import aligned_predictions
    from quiz3.Code.bleu import corpus_bleu
    for row in compared:
        folder=(ROOT/'quiz3/Output/clip_transformer'/row['decoding'] if row['week']=='week5'
                else flickr_root.parents[2]/'quiz4/Output'/row['model']/row['decoding'])
        captions=aligned_predictions(folder/'predictions.json',test)
        require(len(captions)==1000,"Incomplete caption test set")
        bleu=corpus_bleu([r['captions'] for r in test],captions)
        for key in ('bleu1','bleu2','bleu3','bleu4'):near(row[key],bleu[key],key)
        cache=ROOT/'quiz3/Output/bertscore'/f"{row['week']}_{row['model']}_{row['decoding']}"
        with np.load(cache/'per_image.npz') as scores:
            require(list(scores['ids'])==[r['id'] for r in test],"BERTScore IDs disagree")
            for key in ('precision','recall','f1'):
                values=scores[key]
                require(values.shape==(1000,) and np.isfinite(values).all(),"Bad per-image BERTScore")
                near(row['bertscore_'+key],float(values.mean()),key)
        checks.append({'task':'caption','model':row['week']+'/'+row['model']+'/'+row['decoding'],
                       'samples':1000,'bleu4':row['bleu4'],'bertscore_f1':row['bertscore_f1']})
    status={'passed':True,'checks':checks,'scope':'Saved outputs rechecked; no additional training or inference'}
    write_json(ROOT/'analysis/results_audit.json',status)
    print('PASS: retrieval Top-10, classification probabilities/confusion matrices, six full caption evaluations')


if __name__=='__main__':main()
