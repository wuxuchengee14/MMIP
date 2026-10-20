from __future__ import annotations

import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from common.training import amp_context, fit
from common.utils import ROOT, loader, read_json, seed_all, synchronize, worker_seed, write_csv, write_json
from common.vision import BirdImages
from quiz1.Code.prepare import cub_root
from quiz2.Code.features import prepare_text_features
from quiz2.Code.models import FusionClassifier


class CaptionSampler(Sampler):
    def __init__(self, size, captions, seed):
        self.size,self.captions,self.seed,self.epoch = size,captions,seed,0

    def set_epoch(self, epoch):
        self.epoch=epoch

    def __len__(self):return self.size

    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch)
        for index in rng.permutation(self.size):
            yield (int(index),self.epoch,int(rng.integers(self.captions)))


class PairedBirds(BirdImages):
    def __init__(self, root, records, features, train=False, seed=42):
        super().__init__(root,records,train,seed)
        self.features=features

    def __getitem__(self,index):
        slot=index[2] if isinstance(index,tuple) else 0
        image,label=super().__getitem__(index)
        position=index[0] if isinstance(index,tuple) else index
        return image,self.features[position,slot].float(),label


def plot_predictions(root, records, probs, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    predicted=probs.argmax(1)
    correct=[i for i,r in enumerate(records) if predicted[i]==r["label"]]
    wrong=[i for i,r in enumerate(records) if predicted[i]!=r["label"]]
    def diverse(indices, count):
        chosen=[];seen=set()
        for i in indices:
            if records[i]['label'] not in seen:
                chosen.append(i);seen.add(records[i]['label'])
            if len(chosen)==count:break
        return chosen+[i for i in indices if i not in chosen][:count-len(chosen)]
    selected=diverse(correct,3)+diverse(wrong,3)
    fig,axes=plt.subplots(2,3,figsize=(13,9))
    classes=read_json(cub_root()/"classes.json")
    for ax,i in zip(axes.ravel(),selected):
        row=records[i]
        with Image.open(root/row["image"]) as image:ax.imshow(image)
        ax.set_title(f"True: {row['class_name']}\nPred: {classes[predicted[i]]}",fontsize=8)
        import textwrap
        ax.set_xlabel(textwrap.fill(row["captions"][0],45),fontsize=8)
        ax.set_xticks([]);ax.set_yticks([])
    for ax in axes.ravel()[len(selected):]:ax.axis("off")
    fig.tight_layout();fig.savefig(directory/"prediction_examples.png",dpi=130);plt.close(fig)


def save_metrics(labels,probs,classes,directory):
    predicted=probs.argmax(1)
    metrics={"accuracy":float(accuracy_score(labels,predicted)),
             "macro_f1":float(f1_score(labels,predicted,labels=range(len(classes)),average="macro",zero_division=0)),
             "weighted_f1":float(f1_score(labels,predicted,average="weighted",zero_division=0)),"samples":len(labels)}
    write_json(directory/"metrics.json",metrics)
    write_json(directory/"classification_report.json",classification_report(labels,predicted,
               labels=list(range(len(classes))),target_names=classes,output_dict=True,zero_division=0))
    matrix=confusion_matrix(labels,predicted,labels=list(range(len(classes))))
    np.save(directory/"confusion_matrix.npy",matrix)
    errors=[{"true":classes[i],"predicted":classes[j],"count":int(matrix[i,j])}
            for i,j in zip(*np.nonzero(matrix)) if i!=j]
    write_csv(directory/"confusions.csv",sorted(errors,key=lambda r:-r["count"]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,9));view=ax.imshow(matrix,cmap="Blues")
    fig.colorbar(view,ax=ax);ax.set(xlabel="Predicted class index",ylabel="True class index",title="CUB class-index confusion matrix")
    fig.tight_layout();fig.savefig(directory/"confusion_matrix.png",dpi=150);plt.close(fig)
    return metrics


def run(config,device):
    folder=cub_root();manifest=read_json(folder/"manifest.json");root=folder/manifest["image_root"]
    records={s:read_json(folder/f"{s}.json") for s in ("train","val","test")}
    classes=read_json(folder/"classes.json");settings=config["multimodal"]
    text_info=prepare_text_features(config,device)
    caches={s:torch.load(ROOT/f"quiz2/Data/text_features/{s}.pt",weights_only=True) for s in records}
    if any(caches[s]["ids"]!=[r["id"] for r in records[s]] for s in records):raise ValueError("Text cache/image IDs are misaligned")
    comparison=[]
    for kind in settings["backbones"]:
        seed_all(config["seed"])
        model=FusionClassifier(kind,len(classes),text_info["dimension"],settings["fusion_dim"],settings["dropout"]).to(device)
        datasets={s:PairedBirds(root,records[s],caches[s]["features"],s=="train",config["seed"]) for s in records}
        training=DataLoader(datasets["train"],batch_size=settings["batch_size"],
            sampler=CaptionSampler(len(datasets["train"]),caches["train"]["features"].shape[1],config["seed"]),
            num_workers=config["workers"],pin_memory=config["pin_memory"],worker_init_fn=worker_seed,
            persistent_workers=config["workers"]>0)
        validation=loader(datasets["val"],config,settings["batch_size"])
        def step(net,batch,target):
            images,text,labels=[x.to(target,non_blocking=True) for x in batch]
            logits=net(images,text)
            return F.cross_entropy(logits,labels),len(labels),int(logits.argmax(1).eq(labels).sum())
        directory=ROOT/"quiz2/Output"/kind
        done=fit(model,training,validation,step,settings,directory,device,
            {"seed":config["seed"],"backbone":kind,"data":manifest,"text_features":text_info["identity"]})
        model.eval();probabilities=[];synchronize(device);tick=time.perf_counter()
        with torch.inference_mode(),amp_context(device):
            for images,text,_ in loader(datasets["test"],config,settings["batch_size"]):
                logits=model(images.to(device,non_blocking=True),text.to(device,non_blocking=True))
                probabilities.append(logits.float().softmax(1).cpu().numpy())
        synchronize(device);test_seconds=time.perf_counter()-tick
        probs=np.concatenate(probabilities);labels=np.array([r["label"] for r in records["test"]])
        metrics=save_metrics(labels,probs,classes,directory)
        np.savez_compressed(directory/"probabilities.npz",probabilities=probs,labels=labels,ids=[r["id"] for r in records["test"]])
        write_csv(directory/"predictions.csv",[{"id":r["id"],"caption":r["captions"][0],"true_label":r["label"],
            "true_class":r["class_name"],"predicted_label":int(p.argmax()),"predicted_class":classes[p.argmax()],
            "confidence":float(p.max())} for r,p in zip(records["test"],probs)])
        plot_predictions(root,records["test"],probs,directory)
        comparison.append({"image_backbone":kind,"text_backbone":settings["text_model"],**metrics,
            "training_seconds":done["training_seconds"],"test_seconds":test_seconds,
            "trainable_parameters":done["trainable_parameters"],
            "total_parameters_including_frozen_text":done["parameters"]+text_info["parameters"]})
        del model,training,validation,datasets
        if device.type=="cuda":torch.cuda.empty_cache()
    write_json(ROOT/"quiz2/Output/comparison.json",comparison);write_csv(ROOT/"quiz2/Output/comparison.csv",comparison)
    return comparison
