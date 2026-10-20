"""Exercise the three public inference commands using real trained checkpoints."""
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from common.utils import ROOT,file_hash,read_json,write_json


def invoke(task,image,*extra):
    output=ROOT/'analysis/inference'/f'{task}.json'
    command=[sys.executable,str(ROOT/'predict.py'),task,'--image',str(image),'--output',str(output),*extra]
    result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace')
    if result.returncode:raise RuntimeError(result.stdout+'\n'+result.stderr)
    return read_json(output)


def main():
    manifest=read_json(ROOT/'quiz1/Data/cub/manifest.json')
    rows=read_json(ROOT/'quiz1/Data/cub/test.json');mapping={r['id']:r for r in rows}
    gallery=read_json(ROOT/'quiz1/Data/cub/retrieval_splits.json')['test']['gallery']
    row=mapping[gallery[0]];image=ROOT/'quiz1/Data/cub'/manifest['image_root']/row['image']
    retrieved=invoke('retrieval',image,'--backbone','resnet50')
    matches=retrieved['results']
    assert len(matches)==10 and all(file_hash(r['image'])!=file_hash(image) for r in matches)
    assert all(a['similarity']>=b['similarity'] for a,b in zip(matches,matches[1:]))
    row=rows[0];image=ROOT/'quiz1/Data/cub'/manifest['image_root']/row['image']
    classified=invoke('multimodal',image,'--text',row['captions'][0],'--backbone','vit_b_16')
    classes=read_json(ROOT/'quiz1/Data/cub/classes.json')
    with np.load(ROOT/'quiz2/Output/vit_b_16/probabilities.npz') as saved:
        expected=classes[saved['probabilities'][0].argmax()]
    assert classified['predictions'][0]['class']==expected
    config=read_json(ROOT/'config.rtx5090.json');old=(ROOT/config['week4_root']).resolve()
    sample=read_json(old/'quiz1/Data/flickr8k/test.json')[0]
    described=invoke('caption',old/'quiz1/Data/flickr8k'/sample['image'],'--beam-width','3')
    assert described['caption'].strip() and len(described['caption'].split())<=39
    expected=read_json(ROOT/'quiz3/Output/clip_transformer/beam/predictions.json')[0]['caption']
    result={'passed':True,'retrieval_excludes_identical_gallery_image':True,
            'multimodal_matches_saved_top1':True,'caption':described['caption'],
            'caption_matches_batched_prediction':described['caption']==expected,
            'scope':'Three real command-line runs on trained checkpoints; no training'}
    write_json(ROOT/'analysis/inference_probe.json',result)
    print(result)


if __name__=='__main__':main()
