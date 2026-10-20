"""Check Windows spawn, persistent image workers, epoch indices and paired text tensors."""
import gc
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from common.utils import ROOT,worker_seed,write_json
from common.vision import BirdImages
from quiz1.Code.train import ClassBatchSampler
from quiz2.Code.train import CaptionSampler,PairedBirds


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);rows=[]
        for i in range(8):
            Image.fromarray(np.random.default_rng(i).integers(0,255,(48,48,3),dtype=np.uint8)).save(root/f"{i}.jpg")
            rows.append({"id":str(i),"image":f"{i}.jpg","label":i//4})
        sampler=ClassBatchSampler([r["label"] for r in rows],2,2,42)
        batches=DataLoader(BirdImages(root,rows,True,42),batch_sampler=sampler,num_workers=2,
                           persistent_workers=True,worker_init_fn=worker_seed)
        sampler.set_epoch(0);first=torch.cat([x for x,_ in batches])
        sampler.set_epoch(1);second=torch.cat([x for x,_ in batches])
        sampler.set_epoch(0);again=torch.cat([x for x,_ in batches])
        assert torch.equal(first,again) and not torch.equal(first,second)
        del batches;gc.collect()
        features=torch.tensor([[[i*100+j]*6 for j in range(10)] for i in range(8)]).half()
        paired=DataLoader(PairedBirds(root,rows,features,True,42),batch_size=4,sampler=CaptionSampler(8,10,42),
                          num_workers=2,persistent_workers=True,worker_init_fn=worker_seed)
        count=0
        for images,text,labels in paired:
            original_index=text[:,0].long()//100
            assert torch.equal(labels,original_index//4)
            assert images.shape[1:]==(3,224,224) and text.shape[1]==6
            count+=len(labels)
        assert count==8
        del paired;gc.collect()
    write_json(ROOT/"analysis/worker_probe.json",{"pass":True,"workers":2,"persistent":True,
        "epoch_augmentation_reproducible":True,"paired_text_features_aligned":True,"scope":"synthetic loader fixture"})
    print("PASS: spawned persistent workers, epoch-controlled augmentation and paired text feature alignment",flush=True)


if __name__=="__main__":main()
