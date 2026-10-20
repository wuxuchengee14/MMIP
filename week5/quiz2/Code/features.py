"""Frozen DistilBERT text features, retaining all ten training descriptions."""
from __future__ import annotations

import time
import torch
from common.training import amp_context
from common.utils import ROOT, atomic_torch_save, digest, file_hash, read_json, synchronize, write_json
from quiz1.Code.prepare import cub_root


def prepare_text_features(config, device):
    from transformers import AutoModel, AutoTokenizer
    settings = config["multimodal"]
    cache = ROOT / "quiz2/Data/text_features"
    cache.mkdir(parents=True, exist_ok=True)
    data = read_json(cub_root()/"manifest.json")
    identity = digest({"splits": data["split_hashes"], "model": settings["text_model"],
                       "revision":settings.get("text_revision"),
                       "length": settings["text_max_length"], "pooling":"attention-masked mean",
                       "training_captions":"all ten, cycle/select in sampler", "eval_captions":"first only", "version":1})
    marker = cache/"complete.json"
    if marker.exists():
        done = read_json(marker)
        if done["identity"] != identity:
            raise ValueError("Text feature settings/data changed; preserve old cache and use a new directory")
        for split in ("train","val","test"):
            if file_hash(cache/f"{split}.pt") != done["hashes"][split]:
                raise ValueError("Modified text features")
        return done
    options={"revision":settings["text_revision"]} if settings.get("text_revision") else {}
    tokenizer = AutoTokenizer.from_pretrained(settings["text_model"],**options)
    net = AutoModel.from_pretrained(settings["text_model"],**options).to(device).eval().requires_grad_(False)
    tick = time.perf_counter()
    for split in ("train","val","test"):
        rows = read_json(cub_root()/f"{split}.json")
        captions = [r["captions"] if split == "train" else r["captions"][:1] for r in rows]
        texts = [c for group in captions for c in group]
        features = []
        batch_size = settings["text_feature_batch_size"]
        with torch.inference_mode(), amp_context(device):
            for start in range(0,len(texts),batch_size):
                tokens = tokenizer(texts[start:start+batch_size], padding=True, truncation=True,
                                   max_length=settings["text_max_length"],return_tensors="pt").to(device)
                hidden = net(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (hidden.float()*mask).sum(1)/mask.sum(1).clamp_min(1)
                features.append(pooled.half().cpu())
                if start % (batch_size*20)==0:
                    print(f"DistilBERT {split}: {min(start+batch_size,len(texts))}/{len(texts)} descriptions",flush=True)
        tensor = torch.cat(features).reshape(len(rows),len(captions[0]),-1)
        atomic_torch_save({"features":tensor,"ids":[r["id"] for r in rows],"identity":identity},cache/f"{split}.pt")
    synchronize(device)
    done = {"identity":identity,"model":settings["text_model"],"revision":getattr(net.config,"_commit_hash",None),
            "dimension":tensor.shape[-1],"parameters":sum(p.numel() for p in net.parameters()),
            "trainable":False,"pooling":"attention-masked mean","test_caption_policy":"first description per image",
            "extraction_seconds":time.perf_counter()-tick,
            "hashes":{s:file_hash(cache/f"{s}.pt") for s in ("train","val","test")}}
    write_json(marker,done)
    del net
    if device.type=="cuda":torch.cuda.empty_cache()
    return done
