from __future__ import annotations

import math
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler
from common.training import amp_context, fit
from common.utils import ROOT, loader, read_json, seed_all, synchronize, worker_seed, write_csv, write_json
from common.vision import BirdImages
from quiz1.Code.models import RetrievalModel, supervised_contrastive
from quiz1.Code.prepare import cub_root
from quiz1.Code.retrieval import retrieval_metrics


class ClassBatchSampler(Sampler):
    def __init__(self, labels, classes_per_batch, images_per_class, seed):
        self.groups = {label: np.flatnonzero(np.asarray(labels) == label) for label in sorted(set(labels))}
        if classes_per_batch > len(self.groups) or images_per_class < 2:
            raise ValueError("SupCon batch requires P available classes and K >= 2")
        if any(len(v) < images_per_class for v in self.groups.values()):
            raise ValueError("Every sampled class needs K distinct training images")
        self.p, self.k, self.seed, self.epoch = classes_per_batch, images_per_class, seed, 0
        self.steps = math.ceil(len(labels)/(self.p*self.k))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        labels = list(self.groups)
        for _ in range(self.steps):
            chosen = rng.choice(labels, self.p, replace=False)
            yield [(int(i), self.epoch) for label in chosen for i in rng.choice(self.groups[label], self.k, replace=False)]


@torch.inference_mode()
def extract_embeddings(model, batches, device):
    model.eval()
    features = []
    synchronize(device)
    tick = time.perf_counter()
    for images, _ in batches:
        with amp_context(device):
            embedding, _ = model(images.to(device, non_blocking=True))
        features.append(embedding.float().cpu().numpy())
    synchronize(device)
    return np.concatenate(features), time.perf_counter()-tick


def evaluate_embeddings(features, records, split, ks):
    indices = {r["id"]: i for i,r in enumerate(records)}
    q = [indices[sid] for sid in split["query"]]
    g = [indices[sid] for sid in split["gallery"]]
    labels = np.array([r["label"] for r in records])
    return retrieval_metrics(features[q], features[g], labels[q], labels[g],
                             split["query"], split["gallery"], ks)


def plot_retrieval(root, records, split, ranking, ap, directory, count=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mapping = {r["id"]:r for r in records}
    # Include strong and weak queries; all metrics still use every query.
    order = np.argsort(ap, kind="stable")
    chosen=[];seen=set()
    for candidates,limit in ((order,count//2),(order[::-1],count-count//2)):
        if limit<=0:continue
        selected=[]
        for qi in candidates:
            label=mapping[split['query'][qi]]['label']
            if label not in seen and qi not in chosen:
                selected.append(int(qi));seen.add(label)
            if len(selected)==limit:break
        chosen.extend(selected)
    chosen += [int(qi) for qi in order if qi not in chosen][:max(0,count-len(chosen))]
    width = min(10, ranking.shape[1])
    fig, axes = plt.subplots(len(chosen), width+1, figsize=(2*(width+1), 2.5*len(chosen)), squeeze=False)
    for line, qi in enumerate(chosen):
        query = mapping[split["query"][qi]]
        rows = [query] + [mapping[split["gallery"][gi]] for gi in ranking[qi,:width]]
        for col,row in enumerate(rows):
            with Image.open(root / row["image"]) as image:
                axes[line,col].imshow(image)
            axes[line,col].axis("off")
            axes[line,col].set_title(("Query" if col == 0 else f"#{col}") + "\n" + row["class_name"].split(".",1)[-1][:18],
                                    color="black" if col==0 else "green" if row["label"]==query["label"] else "red", fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "retrieval_examples.png", dpi=110)
    plt.close(fig)


def run(config, device):
    folder = cub_root()
    manifest = read_json(folder/"manifest.json")
    root = folder / manifest["image_root"]
    records = {s:read_json(folder/f"{s}.json") for s in ("train","val","test")}
    partitions = read_json(folder/"retrieval_splits.json")
    settings = config["retrieval"]
    comparison = []
    for kind in settings["backbones"]:
        seed_all(config["seed"])
        model = RetrievalModel(kind, manifest["classes"], settings["embedding_dim"]).to(device)
        sampler = ClassBatchSampler([r["label"] for r in records["train"]], settings["classes_per_batch"],
                                    settings["images_per_class"], config["seed"])
        training = DataLoader(BirdImages(root,records["train"],True,config["seed"]), batch_sampler=sampler,
            num_workers=config["workers"], pin_memory=config["pin_memory"], worker_init_fn=worker_seed,
            persistent_workers=config["workers"]>0)
        eval_loaders = {s:loader(BirdImages(root,records[s]),config,settings["eval_batch_size"]) for s in ("val","test")}
        def step(net, batch, target):
            images, labels = [x.to(target, non_blocking=True) for x in batch]
            embedding, logits = net(images)
            loss = supervised_contrastive(embedding, labels, settings["temperature"])
            loss = loss + settings["classification_weight"] * F.cross_entropy(logits, labels)
            return loss, len(labels), int(logits.argmax(1).eq(labels).sum())
        def validate(net, target):
            features, _ = extract_embeddings(net, eval_loaders["val"], target)
            metrics, *_ = evaluate_embeddings(features, records["val"], partitions["val"], settings["top_k"])
            return {"val_loss": 1-metrics["map_at_10"], "val_accuracy": metrics["map_at_10"]}
        directory = ROOT / "quiz1/Output" / kind
        done = fit(model, training, None, step, settings, directory, device,
                   {"seed":config["seed"],"backbone":kind,"data":manifest,"task":"SupCon+CE retrieval",
                    "selection_metric":"validation_mAP@10","accuracy_labels":["train classifier Top-1","validation mAP@10"]}, validate=validate)
        features, extraction_seconds = extract_embeddings(model, eval_loaders["test"], device)
        tick = time.perf_counter()
        metrics, ranking, similarity, ap = evaluate_embeddings(features, records["test"], partitions["test"], settings["top_k"])
        metrics.update(embedding_seconds=extraction_seconds, search_seconds=time.perf_counter()-tick,
                       parameters=done["parameters"], training_seconds=done["training_seconds"])
        write_json(directory/"metrics.json", metrics)
        np.savez_compressed(directory/"embeddings.npz", features=features, ids=[r["id"] for r in records["test"]])
        mapping = {r["id"]:r for r in records["test"]}
        predictions = []
        for qi,sid in enumerate(partitions["test"]["query"]):
            for rank,gi in enumerate(ranking[qi,:max(settings["top_k"])],1):
                gid = partitions["test"]["gallery"][int(gi)]
                predictions.append({"query_id":sid,"gallery_id":gid,"rank":rank,
                    "similarity":float(similarity[qi,gi]),"relevant":mapping[sid]["label"]==mapping[gid]["label"]})
        write_csv(directory/"retrieval_results.csv", predictions)
        plot_retrieval(root,records["test"],partitions["test"],ranking,ap,directory,settings["example_queries"])
        comparison.append({"backbone":kind,**metrics})
        del model,training,eval_loaders
        if device.type=="cuda": torch.cuda.empty_cache()
    write_json(ROOT/"quiz1/Output/comparison.json",comparison)
    write_csv(ROOT/"quiz1/Output/comparison.csv",comparison)
    return comparison
