from __future__ import annotations

import sys
import time
import textwrap
from pathlib import Path
if __package__ in (None, ""):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset
from common.text import encode
from common.training import amp_context,fit
from common.utils import ROOT,digest,loader,read_json,seed_all,synchronize,write_csv,write_json
from quiz1.Code.prepare import week4_root
from quiz3.Code.models import CaptionModel,decode,generate
from quiz3.Code.bleu import corpus_bleu
from quiz3.Code.features import prepare_features,benchmark_encoder


class CaptionDataset(Dataset):
    def __init__(self,features,records,vocab,length):
        self.features=features;words={word:i for i,word in enumerate(vocab)}
        self.samples=[]
        for index,row in enumerate(records):
            for caption in row["captions"]:
                tokens=encode(caption,words,length,True)
                self.samples.append((index,tokens+[0]*(length-len(tokens))))

    def __len__(self):return len(self.samples)

    def __getitem__(self,index):
        image,tokens=self.samples[index]
        return self.features[image].float(),torch.tensor(tokens)


def step(model,batch,device):
    features,tokens=[x.to(device,non_blocking=True) for x in batch]
    logits=model(features,tokens[:,:-1]);target=tokens[:,1:];valid=target.ne(0)
    loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),ignore_index=0)
    return loss,int(valid.sum()),int((logits.argmax(-1).eq(target)&valid).sum())


@torch.inference_mode()
def predict(model,features,vocab,settings,device,width):
    model.eval();results=[]
    with amp_context(device):
        for start in range(0,len(features),settings["decode_batch_size"]):
            tokens=generate(model,features[start:start+settings["decode_batch_size"]].to(device).float(),
                            settings["max_length"],width,settings["beam_length_penalty"])
            results.extend(decode(row,vocab) for row in tokens.cpu().tolist())
    return results


@torch.inference_mode()
def benchmark(model,features,settings,device,width):
    model.eval();times=[];counts=[]
    with amp_context(device):
        first=features[:1].to(device).float()
        for _ in range(3):generate(model,first,settings["max_length"],width,settings["beam_length_penalty"])
        for _ in range(settings["timing_repeats"]):
            for i in range(min(settings["timing_samples"],len(features))):
                feature=features[i:i+1].to(device).float();synchronize(device);tick=time.perf_counter()
                tokens=generate(model,feature,settings["max_length"],width,settings["beam_length_penalty"])
                synchronize(device);times.append(time.perf_counter()-tick)
                counts.append(sum(t not in (0,2,3) for t in tokens[0].tolist()))
    return {"decoder_mean_ms":float(np.mean(times)*1000),"decoder_p95_ms":float(np.percentile(times,95)*1000),
        "mean_generated_tokens":float(np.mean(counts)),"batch_size":1,
        "timing_images":min(settings["timing_samples"],len(features)),"timing_repeats":settings["timing_repeats"]}


def plot_examples(folder,records,captions,directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(13,15))
    for ax,row,caption in zip(axes.ravel(),records[:6],captions[:6]):
        with Image.open(folder/row["image"]) as image:ax.imshow(image)
        ax.axis("off");ax.set_title(textwrap.fill("CLIP generated: "+caption,55)+"\n"+
                                    textwrap.fill("Reference: "+row["captions"][0],55),fontsize=9)
    fig.tight_layout();fig.savefig(directory/"caption_examples.png",dpi=130);plt.close(fig)


def run(config,device):
    folder=week4_root(config)/"quiz1/Data/flickr8k";settings=config["caption"]
    old_settings=read_json(week4_root(config)/"config.rtx5090.json")["caption"]
    matching=("dim","layers","heads","dropout","max_length","grid_size","beam_width","beam_length_penalty")
    if any(settings[k]!=old_settings[k] for k in matching):
        raise ValueError("Match Week4 decoder and decoding settings before making the controlled comparison")
    vocab=read_json(folder/"vocab.json")
    if vocab[:4]!=["<pad>","<unk>","<bos>","<eos>"]:raise ValueError("Unexpected Week4 special tokens")
    records={s:read_json(folder/f"{s}.json") for s in ("train","val","test")}
    feature_info=prepare_features(config,device)
    caches={s:torch.load(ROOT/f"quiz3/Data/clip_features/{s}.pt",weights_only=True) for s in records}
    if any(caches[s]["ids"]!=[r["id"] for r in records[s]] for s in records):raise ValueError("CLIP cache/record IDs mismatch")
    feature_config=dict(config,workers=0)
    batches={s:loader(CaptionDataset(caches[s]["features"],records[s],vocab,settings["max_length"]),
                     feature_config,settings["batch_size"],shuffle=s=="train") for s in ("train","val")}
    seed_all(config["seed"])
    model=CaptionModel("transformer",len(vocab),**{k:settings[k] for k in ("dim","layers","heads","dropout","max_length","grid_size")},
                       feature_dim=feature_info["feature_shape"][1]).to(device)
    directory=ROOT/"quiz3/Output/clip_transformer"
    done=fit(model,batches["train"],batches["val"],step,settings,directory,device,
             {"seed":config["seed"],"features":feature_info["identity"],"vocab":digest(vocab),"model":"CLIP+Transformer"},maximize=False)
    encoder_timing=benchmark_encoder(config,records["test"],device)
    write_json(ROOT/"quiz3/Output/encoder_timing.json",encoder_timing)
    comparison=[]
    for mode,width in (("greedy",1),("beam",settings["beam_width"])):
        captions=predict(model,caches["test"]["features"],vocab,settings,device,width)
        metrics=corpus_bleu([r["captions"] for r in records["test"]],captions)
        timing=benchmark(model,caches["test"]["features"],settings,device,width)
        timing["estimated_pipeline_mean_ms"]=timing["decoder_mean_ms"]+encoder_timing["encoder_mean_ms"]
        timing["note"]="Sum of separately measured encoder/decoder means; not per-image end-to-end latency"
        output=directory/mode;output.mkdir(parents=True,exist_ok=True)
        rows=[{"id":r["id"],"image":r["image"],"caption":c,"references":r["captions"]} for r,c in zip(records["test"],captions)]
        write_json(output/"predictions.json",rows);write_json(output/"bleu.json",metrics);write_json(output/"timing.json",timing)
        write_csv(output/"predictions.csv",[{"id":r["id"],"caption":r["caption"]} for r in rows])
        plot_examples(folder,records["test"],captions,output)
        comparison.append({"source":"week5","model":"clip_transformer","decoding":mode,**metrics,**timing,
             "decoder_parameters":done["parameters"],"frozen_encoder_parameters":feature_info["encoder_parameters"],
             "training_seconds":done["training_seconds"]})
    write_json(ROOT/"quiz3/Output/clip_results.json",comparison)
    del model
    if device.type=="cuda":torch.cuda.empty_cache()
    return comparison
