from __future__ import annotations

import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import models

from common.training import amp_context
from common.utils import ROOT, atomic_torch_save, digest, file_hash, loader, read_json, synchronize, write_json


class ImageDataset(Dataset):
    def __init__(self, root, records):
        self.root, self.records = Path(root), records
        self.transform = models.ResNet50_Weights.IMAGENET1K_V1.transforms()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with Image.open(self.root / self.records[index]["image"]) as image:
            return self.transform(image.convert("RGB"))


def encoder(grid_size=4, pretrained=True):
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    model = nn.Sequential(*list(resnet.children())[:-2], nn.AdaptiveAvgPool2d((grid_size, grid_size)))
    model.requires_grad_(False)
    return model.eval()


def feature_tokens(model, images):
    return model(images).flatten(2).transpose(1, 2)


def prepare_features(config, device):
    folder = ROOT / "quiz1" / "Data" / "flickr8k"
    cache = ROOT / "quiz4" / "Data" / "features"
    cache.mkdir(parents=True, exist_ok=True)
    manifest = read_json(folder / "manifest.json")
    settings = config["caption"]
    identity = digest({"data": manifest, "weights": "ResNet50_IMAGENET1K_V1",
                       "grid": settings["grid_size"], "version": 1})
    ready = cache / "complete.json"
    if ready.exists():
        old = read_json(ready)
        if old["identity"] != identity:
            raise RuntimeError("Caption feature configuration/data changed; use a fresh feature cache.")
        if any(file_hash(cache / f"{s}.pt") != old["hashes"][s] for s in ("train", "val", "test")):
            raise ValueError("Modified caption feature cache")
        return old
    net = encoder(settings["grid_size"]).to(device)
    counts = {}
    synchronize(device)
    tick = time.perf_counter()
    for split in ("train", "val", "test"):
        records = read_json(folder / f"{split}.json")
        batches = loader(ImageDataset(folder, records), config, settings["feature_batch_size"])
        features = []
        with torch.inference_mode(), amp_context(device):
            for i, images in enumerate(batches):
                features.append(feature_tokens(net, images.to(device, non_blocking=True)).half().cpu())
                if i % 10 == 0:
                    print(f"ResNet features {split}: batch {i+1}/{len(batches)}", flush=True)
        tensor = torch.cat(features)
        atomic_torch_save({"features": tensor, "ids": [r["id"] for r in records], "identity": identity},
                          cache / f"{split}.pt")
        counts[split] = len(tensor)
        del features, tensor
    synchronize(device)
    done = {"identity": identity, "counts": counts, "extraction_seconds": time.perf_counter() - tick,
            "feature_shape": [settings["grid_size"]**2, 2048], "dtype": "float16",
            "weights": "ResNet50_IMAGENET1K_V1", "encoder_parameters": sum(p.numel() for p in net.parameters()),
            "hashes": {s: file_hash(cache / f"{s}.pt") for s in counts}}
    write_json(ready, done)
    del net
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return done
