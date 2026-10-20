"""Frozen CLIP patch tokens pooled to the Week4 decoder's 4x4 spatial grid."""
from __future__ import annotations

import math
import time
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from common.training import amp_context
from common.utils import ROOT, atomic_torch_save, digest, file_hash, loader, read_json, synchronize, write_json
from quiz1.Code.prepare import week4_root


class ClipImages(Dataset):
    def __init__(self, root, records, processor):
        self.root,self.records=root,records
        size=processor.crop_size["height"]
        self.transform=transforms.Compose([transforms.Resize(processor.size["shortest_edge"],
            interpolation=transforms.InterpolationMode.BICUBIC),transforms.CenterCrop(size),
            transforms.ToTensor(),transforms.Normalize(processor.image_mean,processor.image_std)])

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        with Image.open(self.root/self.records[index]["image"]) as image:
            return self.transform(image.convert("RGB"))


def patch_features(net,pixels,grid):
    tokens=net(pixel_values=pixels).last_hidden_state[:,1:]
    # CLIP's post layer norm is ordinarily applied to CLS only; apply it to each patch.
    tokens=net.vision_model.post_layernorm(tokens)
    side=math.isqrt(tokens.shape[1])
    if side*side!=tokens.shape[1]:raise ValueError("CLIP patch token grid is not square")
    spatial=tokens.transpose(1,2).reshape(len(tokens),tokens.shape[2],side,side)
    return F.adaptive_avg_pool2d(spatial,(grid,grid)).flatten(2).transpose(1,2)


def load_encoder(config,device):
    from transformers import CLIPImageProcessor, CLIPVisionModel
    from huggingface_hub import try_to_load_from_cache
    from pathlib import Path
    name=config["caption"]["clip_model"]
    revision=config["caption"].get("clip_revision")
    options={"revision":revision} if revision else {}
    processor=CLIPImageProcessor.from_pretrained(name,**options)
    net=CLIPVisionModel.from_pretrained(name,**options).eval().requires_grad_(False).to(device)
    cached=try_to_load_from_cache(name,"config.json",revision=revision or "main")
    net.mmip_revision=Path(cached).parent.name if isinstance(cached,str) else getattr(net.config,"_commit_hash",None)
    return net,processor


def prepare_features(config,device):
    settings=config["caption"]
    folder=week4_root(config)/"quiz1/Data/flickr8k"
    manifest=read_json(ROOT/"quiz3/Data/flickr_manifest.json")
    identity=digest({"data":manifest,"model":settings["clip_model"],"revision":settings.get("clip_revision"),"grid":settings["grid_size"],
                     "tokens":"exclude CLS; post-layernorm each patch; adaptive 4x4 average pooling","version":1})
    cache=ROOT/"quiz3/Data/clip_features";cache.mkdir(parents=True,exist_ok=True)
    marker=cache/"complete.json"
    if marker.exists():
        done=read_json(marker)
        if done["identity"]!=identity:raise ValueError("CLIP data/model/grid changed; use a fresh feature cache")
        if any(file_hash(cache/f"{s}.pt")!=done["hashes"][s] for s in ("train","val","test")):
            raise ValueError("Modified CLIP feature cache")
        return done
    net,processor=load_encoder(config,device)
    synchronize(device);tick=time.perf_counter()
    counts={}
    for split in ("train","val","test"):
        rows=read_json(folder/f"{split}.json")
        features=[]
        batches=loader(ClipImages(folder,rows,processor),config,settings["feature_batch_size"])
        with torch.inference_mode(),amp_context(device):
            for i,pixels in enumerate(batches):
                features.append(patch_features(net,pixels.to(device,non_blocking=True),settings["grid_size"]).half().cpu())
                if i%10==0:print(f"CLIP features {split}: batch {i+1}/{len(batches)}",flush=True)
        tensor=torch.cat(features)
        atomic_torch_save({"features":tensor,"ids":[r["id"] for r in rows],"identity":identity},cache/f"{split}.pt")
        counts[split]=len(rows)
    synchronize(device)
    done={"identity":identity,"counts":counts,"model":settings["clip_model"],
          "revision":net.mmip_revision,"processor":processor.to_dict(),
          "feature_shape":[settings["grid_size"]**2,tensor.shape[-1]],
          "extraction_seconds":time.perf_counter()-tick,"encoder_frozen":True,
          "encoder_parameters":sum(p.numel() for p in net.parameters()),
          "hashes":{s:file_hash(cache/f"{s}.pt") for s in counts}}
    write_json(marker,done)
    del net
    if device.type=="cuda":torch.cuda.empty_cache()
    return done


def benchmark_encoder(config,records,device):
    net,processor=load_encoder(config,device)
    dataset=ClipImages(week4_root(config)/"quiz1/Data/flickr8k",records,processor)
    settings=config["caption"];times=[]
    with torch.inference_mode(),amp_context(device):
        first=dataset[0][None].to(device)
        for _ in range(3):patch_features(net,first,settings["grid_size"])
        for _ in range(settings["timing_repeats"]):
            for i in range(min(settings["timing_samples"],len(dataset))):
                synchronize(device);tick=time.perf_counter()
                patch_features(net,dataset[i][None].to(device),settings["grid_size"])
                synchronize(device);times.append(time.perf_counter()-tick)
    del net
    return {"encoder_mean_ms":sum(times)/len(times)*1000,"batch_size":1,"samples":min(settings["timing_samples"],len(dataset)),
            "repeats":settings["timing_repeats"],"includes":"image read, CLIP preprocessing, transfer, vision forward and pooling"}
