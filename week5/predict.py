"""Inference on a user image using trained Week5 checkpoints."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from common.training import amp_context
from common.utils import ROOT,configure,read_json,write_json
from common.vision import BirdImages
from quiz1.Code.prepare import cub_root,week4_root


def image_tensor(path,device):
    return BirdImages(path.parent,[{"image":path.name,"label":0}])[0][0][None].to(device)


def retrieve(args,config,device):
    from quiz1.Code.models import RetrievalModel
    directory=ROOT/"quiz1/Output"/args.backbone
    done=read_json(directory/"complete.json");settings=done["settings"]
    manifest=read_json(cub_root()/"manifest.json")
    if done["identity"]["data"]!=manifest:raise ValueError("Checkpoint and CUB dataset identity disagree")
    model=RetrievalModel(args.backbone,manifest["classes"],settings["embedding_dim"],pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(directory/"best.pt",map_location=device,weights_only=True)["model"])
    with torch.inference_mode(),amp_context(device):feature,_=model(image_tensor(args.image,device))
    records=read_json(cub_root()/"test.json");mapping={r["id"]:r for r in records}
    partition=read_json(cub_root()/"retrieval_splits.json")["test"]["gallery"]
    from common.utils import file_hash
    query_hash=file_hash(args.image)
    eligible=[sid for sid in partition if mapping[sid]["image_sha256"]!=query_hash]
    if not eligible:raise ValueError("No gallery images remain after excluding the query image")
    with np.load(directory/"embeddings.npz") as cache:
        indices={str(sid):i for i,sid in enumerate(cache["ids"])}
        features=cache["features"][[indices[sid] for sid in eligible]]
    features=features/np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-12)
    similarity=features@feature[0].float().cpu().numpy()
    chosen=np.argsort(-similarity,kind="stable")[:args.top_k]
    root=cub_root()/manifest["image_root"]
    return {"task":"retrieval","backbone":args.backbone,"query":str(args.image),
        "gallery_scope":"official test gallery; query file hash excluded", "results":[
            {"rank":rank,"id":eligible[i],"class":mapping[eligible[i]]["class_name"],
             "similarity":float(similarity[i]),"image":str(root/mapping[eligible[i]]["image"])}
            for rank,i in enumerate(chosen,1)]}


def classify(args,config,device):
    from transformers import AutoModel,AutoTokenizer
    from quiz2.Code.models import FusionClassifier
    directory=ROOT/"quiz2/Output"/args.backbone;done=read_json(directory/"complete.json")
    settings=done["settings"];text_info=read_json(ROOT/"quiz2/Data/text_features/complete.json")
    if done["identity"]["text_features"]!=text_info["identity"]:raise ValueError("Text cache and checkpoint identity disagree")
    classes=read_json(cub_root()/"classes.json")
    options={"revision":text_info["revision"]} if text_info.get("revision") else {}
    tokenizer=AutoTokenizer.from_pretrained(settings["text_model"],**options)
    encoder=AutoModel.from_pretrained(settings["text_model"],**options).to(device).eval()
    tokens=tokenizer([args.text],return_tensors="pt",truncation=True,max_length=settings["text_max_length"]).to(device)
    model=FusionClassifier(args.backbone,len(classes),text_info["dimension"],settings["fusion_dim"],settings["dropout"],False).to(device).eval()
    model.load_state_dict(torch.load(directory/"best.pt",map_location=device,weights_only=True)["model"])
    with torch.inference_mode(),amp_context(device):
        hidden=encoder(**tokens).last_hidden_state;mask=tokens["attention_mask"].unsqueeze(-1)
        text=((hidden.float()*mask).sum(1)/mask.sum(1).clamp_min(1)).half().float()
        probabilities=model(image_tensor(args.image,device),text).float().softmax(1)[0]
    scores,labels=probabilities.topk(min(5,len(classes)))
    return {"task":"multimodal classification","image":str(args.image),"text":args.text,"backbone":args.backbone,
        "predictions":[{"class":classes[int(label)],"probability":float(score)} for score,label in zip(scores,labels)]}


def describe(args,config,device):
    from quiz3.Code.features import ClipImages,load_encoder,patch_features
    from quiz3.Code.models import CaptionModel,decode,generate
    directory=ROOT/"quiz3/Output/clip_transformer";done=read_json(directory/"complete.json")
    settings=done["settings"];meta=read_json(ROOT/"quiz3/Data/clip_features/complete.json")
    if done["identity"]["features"]!=meta["identity"]:raise ValueError("CLIP cache and checkpoint identity disagree")
    vocab=read_json(week4_root(config)/"quiz1/Data/flickr8k/vocab.json")
    from common.utils import digest
    if digest(vocab)!=done["identity"]["vocab"]:raise ValueError("Changed caption vocabulary")
    encoder,processor=load_encoder(dict(config,caption=dict(settings,clip_revision=meta.get("revision"))),device)
    pixels=ClipImages(args.image.parent,[{"image":args.image.name}],processor)[0][None].to(device)
    model=CaptionModel("transformer",len(vocab),**{k:settings[k] for k in ("dim","layers","heads","dropout","max_length","grid_size")},
                       feature_dim=meta["feature_shape"][1]).to(device).eval()
    model.load_state_dict(torch.load(directory/"best.pt",map_location=device,weights_only=True)["model"])
    with torch.inference_mode(),amp_context(device):
        features=patch_features(encoder,pixels,settings["grid_size"])
        tokens=generate(model,features,settings["max_length"],args.beam_width,settings["beam_length_penalty"])
    return {"task":"CLIP image captioning","image":str(args.image),"beam_width":args.beam_width,
            "caption":decode(tokens[0].cpu().tolist(),vocab)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task",choices=("retrieval","multimodal","caption"))
    parser.add_argument("--image",required=True,type=Path)
    parser.add_argument("--text",default="")
    parser.add_argument("--backbone",choices=("resnet50","vit_b_16"),default="resnet50")
    parser.add_argument("--top-k",type=int,default=10)
    parser.add_argument("--beam-width",type=int,default=3)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--config",type=Path,default=ROOT/"config.rtx5090.json")
    parser.add_argument("--output",type=Path,default=ROOT/"inference/result.json")
    args=parser.parse_args();args.image=args.image.resolve()
    if not args.image.is_file():parser.error("Image file not found")
    if args.task=="multimodal" and not args.text.strip():parser.error("Multimodal classification requires --text")
    if args.top_k<1 or args.beam_width<1:parser.error("top-k and beam-width must be positive")
    config=read_json(args.config);device=configure(config,args.device)
    result={"retrieval":retrieve,"multimodal":classify,"caption":describe}[args.task](args,config,device)
    write_json(args.output,result)
    import json
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
