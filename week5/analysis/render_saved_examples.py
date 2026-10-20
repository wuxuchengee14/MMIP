"""Render diverse examples from saved predictions without retraining any model."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from common.utils import ROOT,read_json
from quiz1.Code.prepare import cub_root
from quiz1.Code.train import evaluate_embeddings,plot_retrieval
from quiz2.Code.train import plot_predictions


def main():
    config=read_json(ROOT/'config.rtx5090.json')
    records=read_json(cub_root()/'test.json')
    root=cub_root()/read_json(cub_root()/'manifest.json')['image_root']
    split=read_json(cub_root()/'retrieval_splits.json')['test']
    for kind in config['retrieval']['backbones']:
        folder=ROOT/'quiz1/Output'/kind
        with np.load(folder/'embeddings.npz') as cache:features=cache['features']
        _,ranking,_,ap=evaluate_embeddings(features,records,split,config['retrieval']['top_k'])
        plot_retrieval(root,records,split,ranking,ap,folder,config['retrieval']['example_queries'])
    for kind in config['multimodal']['backbones']:
        folder=ROOT/'quiz2/Output'/kind
        with np.load(folder/'probabilities.npz') as cache:probs=cache['probabilities']
        plot_predictions(root,records,probs,folder)
    print('Rendered CNN / ViT examples from saved predictions; metrics and weights unchanged')


if __name__=='__main__':main()
