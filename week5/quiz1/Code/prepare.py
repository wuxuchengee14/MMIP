"""Prepare shared CUB image/caption splits; preserve the official image test split."""
from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from sklearn.model_selection import train_test_split
from common.data import download, extract
from common.utils import ROOT, digest, file_hash, read_json, table, write_json


def week4_root(config):
    return (ROOT / config["week4_root"]).resolve()


def cub_root():
    return ROOT / "quiz1/Data/cub"


def keyed(path):
    return dict(line.strip().split(maxsplit=1) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def divide_query_gallery(rows, seed):
    query, gallery = [], []
    rng = np.random.default_rng(seed)
    for label in sorted({r["label"] for r in rows}):
        group = [r for r in rows if r["label"] == label]
        if len(group) < 2:
            raise ValueError("Every retrieval class requires at least two images")
        order = rng.permutation(len(group))
        cut = max(1, len(group)//2)
        query.extend(group[i]["id"] for i in order[:cut])
        gallery.extend(group[i]["id"] for i in order[cut:])
    return {"query": sorted(query), "gallery": sorted(gallery)}


def parse_cub(image_folder, caption_folder, settings, seed):
    names = keyed(image_folder / "images.txt")
    labels = keyed(image_folder / "image_class_labels.txt")
    flags = keyed(image_folder / "train_test_split.txt")
    classes = keyed(image_folder / "classes.txt")
    if not (set(names) == set(labels) == set(flags)):
        raise ValueError("CUB image/label/split metadata do not align")
    if len(names) != settings["expected_images"] or len(classes) != settings["expected_classes"]:
        raise ValueError("Unexpected CUB image/class counts")
    records = []
    for sid in sorted(names, key=int):
        relative = names[sid]
        text = caption_folder / Path(relative).with_suffix(".txt")
        if not text.is_file():
            raise FileNotFoundError(f"Missing human captions: {text}. Raw text captions, not pretrained embeddings, are required.")
        captions = [re.sub(r"\s+", " ", line).strip() for line in text.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(captions) != settings["captions_per_image"]:
            raise ValueError(f"Expected ten human captions for {relative}; found {len(captions)}")
        image = image_folder / "images" / relative
        if not image.is_file():
            raise FileNotFoundError(image)
        records.append({"id": sid, "image": "images/" + relative, "label": int(labels[sid])-1,
            "class_name": classes[labels[sid]], "captions": captions, "image_sha256": file_hash(image),
            "official_train": flags[sid] == "1"})
    official_train = [r for r in records if r["official_train"]]
    test = [r for r in records if not r["official_train"]]
    train, val = train_test_split(official_train, test_size=settings["validation_fraction"], random_state=seed,
                                 stratify=[r["label"] for r in official_train])
    splits = {s: sorted(rows, key=lambda r: int(r["id"])) for s, rows in (("train", train), ("val", val), ("test", test))}
    # Identical pixels/files should not make a query match itself across evaluation partitions.
    hashes = {s: {r["image_sha256"] for r in rows} for s, rows in splits.items()}
    overlaps = {f"{a}/{b}": len(hashes[a] & hashes[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    if any(overlaps.values()):
        raise ValueError(f"Exact duplicate images across splits: {overlaps}")
    return splits, [classes[str(i+1)] for i in range(len(classes))]


def obtain_captions(config):
    settings = config["data"]
    folder = cub_root()
    explicit = os.environ.get("CUB_CAPTIONS_DIR", settings["captions_directory"])
    if explicit:
        return Path(explicit).expanduser().resolve()
    expected = folder / "captions/birds/text"
    if expected.is_dir() and len(list(expected.glob("*/*.txt")))==settings["expected_images"]:
        return expected
    archive = folder / "downloads/birds.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    local_archive = os.environ.get("CUB_CAPTIONS_ARCHIVE")
    if local_archive:
        archive = Path(local_archive).resolve()
    elif settings["captions_url"]:
        download(settings["captions_url"], archive)
    elif not archive.exists():
        from common.network import enable_system_trust
        enable_system_trust()
        import gdown
        partial = archive.with_name("birds.zip.part")
        try:
            result = gdown.download(id=settings["captions_drive_id"], output=str(partial), quiet=False)
            if not result or not zipfile.is_zipfile(partial):
                raise IOError("Google Drive did not return a valid ZIP archive")
            partial.replace(archive)
        except Exception as error:
            raise RuntimeError("Could not download AttnGAN's original birds metadata. Download birds.zip from "
                "https://github.com/taoxugit/AttnGAN and set CUB_CAPTIONS_ARCHIVE, or point "
                "CUB_CAPTIONS_DIR to the extracted text/ directory. No captions will be fabricated.") from error
    extract(archive, folder / "captions")
    nested = list((folder / "captions").rglob("text.zip"))
    for path in nested:
        extract(path, path.parent)
    candidates = [p for p in (folder / "captions").rglob("text")
                  if p.is_dir() and "__MACOSX" not in p.parts and any(p.rglob("*.txt"))]
    if len(candidates) != 1:
        raise ValueError("Cannot uniquely locate original CUB text/ directory; set CUB_CAPTIONS_DIR explicitly")
    return candidates[0]


def prepare_cub(config):
    folder = cub_root()
    folder.mkdir(parents=True, exist_ok=True)
    signature = digest({"seed": config["seed"], "validation_fraction": config["data"]["validation_fraction"], "version": 1})
    ready = folder / "manifest.json"
    if ready.exists():
        manifest = read_json(ready)
        if manifest["signature"] != signature:
            raise RuntimeError("CUB split settings changed; preserve the old directory and prepare a new dataset")
        for split in ("train", "val", "test"):
            if file_hash(folder / f"{split}.json") != manifest["split_hashes"][split]:
                raise ValueError("CUB split file modified")
        if (file_hash(folder/"retrieval_splits.json")!=manifest["retrieval_splits_sha256"] or
                file_hash(folder/"classes.json")!=manifest["classes_sha256"]):
            raise ValueError("CUB query/gallery or class metadata modified")
        return manifest
    images = folder / "raw/CUB_200_2011"
    extraction_marker=folder/"raw/extraction_complete.json"
    if not extraction_marker.exists():
        archive = download(config["data"]["cub_images_url"], folder / "downloads/CUB_200_2011.tgz", config["data"]["cub_images_md5"],
                           workers=config["data"].get("download_workers",4))
        extract(archive, folder / "raw")
        write_json(extraction_marker,{"source_md5":config["data"]["cub_images_md5"]})
    captions = obtain_captions(config)
    splits, classes = parse_cub(images, captions, config["data"], config["seed"])
    for split, rows in splits.items():
        write_json(folder / f"{split}.json", rows)
    write_json(folder / "classes.json", classes)
    retrieval = {s: divide_query_gallery(splits[s], config["seed"]) for s in ("val", "test")}
    write_json(folder / "retrieval_splits.json", retrieval)
    manifest = {"signature": signature, "counts": {s: len(rows) for s, rows in splits.items()},
        "classes": len(classes), "captions_per_image": 10, "image_root": "raw/CUB_200_2011",
        "source": "https://www.vision.caltech.edu/datasets/cub_200_2011/",
        "text_origin": "Reed et al. CVPR 2016 human descriptions, distributed in AttnGAN birds metadata",
        "text_source": "https://github.com/taoxugit/AttnGAN", "caption_root": str(captions),
        "split_hashes": {s: file_hash(folder / f"{s}.json") for s in splits},
        "retrieval_splits_sha256": file_hash(folder / "retrieval_splits.json"),
        "classes_sha256": file_hash(folder / "classes.json"),
        "limitation": "CUB officially warns of ImageNet/Flickr pretraining overlap; no claim of pretraining-independent generalization"}
    write_json(ready, manifest)
    return manifest


def prepare_flickr(config):
    original = week4_root(config) / "quiz1/Data/flickr8k"
    marker=ROOT/"quiz3/Data/flickr_manifest.json"
    split_hashes={s:file_hash(original/f"{s}.json") for s in ("train","val","test")}
    vocab_hash=file_hash(original/"vocab.json")
    if marker.exists():
        previous=read_json(marker)
        if (previous.get("source_directory")==str(original) and previous.get("split_hashes")==split_hashes
                and previous.get("vocab_sha256")==vocab_hash and len(previous.get("image_hashes",{}))==8000):
            return previous
    rows = {s: read_json(original / f"{s}.json") for s in ("train", "val", "test")}
    ids = {s: {r["id"] for r in values} for s, values in rows.items()}
    if any(ids[a] & ids[b] for a,b in (("train","val"),("train","test"),("val","test"))):
        raise ValueError("Flickr images overlap across splits")
    for split, values in rows.items():
        if len(values) != {"train": 6000, "val": 1000, "test": 1000}[split] or len(ids[split]) != len(values):
            raise ValueError("Expected original full Week4 Flickr8k splits")
        if any(len(r["captions"]) != 5 or not (original / r["image"]).is_file() for r in values):
            raise ValueError("Missing Flickr originals or five-reference captions; prepare Week4 data first")
    manifest = {"source_directory": str(original), "counts": {s:len(v) for s,v in rows.items()},
        "split_hashes": split_hashes,
        "image_hashes": {r["id"]:file_hash(original/r["image"]) for values in rows.values() for r in values},
        "vocab_sha256": vocab_hash, "week4_data_read_only": True}
    write_json(ROOT / "quiz3/Data/flickr_manifest.json", manifest)
    return manifest


def run(config):
    cub = prepare_cub(config)
    rows={s:read_json(cub_root()/f"{s}.json") for s in ("train","val","test")}
    partitions=read_json(cub_root()/"retrieval_splits.json")
    audit={}
    for split,values in rows.items():
        counts=Counter(r["image_sha256"] for r in values)
        audit[split]={"images":len(values),"classes":len({r["label"] for r in values}),
                      "duplicate_hash_groups":sum(n>1 for n in counts.values())}
        if split!="train":
            hashes={r["id"]:r["image_sha256"] for r in values}
            part=partitions[split]
            overlap={hashes[s] for s in part["query"]}&{hashes[s] for s in part["gallery"]}
            if overlap:raise ValueError("Exact duplicate images between retrieval query and gallery")
            audit[split].update(query=len(part["query"]),gallery=len(part["gallery"]),query_gallery_hash_overlap=0)
    write_json(ROOT/"quiz1/Output/data_audit.json",audit)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import textwrap
    chosen=[];seen=set()
    for row in rows["train"]:
        if row["label"] not in seen:chosen.append(row);seen.add(row["label"])
        if len(chosen)==6:break
    fig,axes=plt.subplots(2,3,figsize=(14,10))
    for ax,row in zip(axes.ravel(),chosen):
        with Image.open(cub_root()/cub["image_root"]/row["image"]) as image:ax.imshow(image)
        ax.set_title(row["class_name"],fontsize=9)
        ax.set_xlabel(textwrap.fill(row["captions"][0],48),fontsize=8)
        ax.set_xticks([]);ax.set_yticks([])
    fig.tight_layout();fig.savefig(ROOT/"quiz1/Output/cub_examples.png",dpi=130);plt.close(fig)
    flickr = prepare_flickr(config)
    write_json(ROOT / "quiz1/Output/datasets.json", {"cub": cub, "flickr": flickr})
    print("Prepared CUB", cub["counts"], "and reused Flickr", flickr["counts"], flush=True)
    return cub, flickr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.rtx5090.json")
    run(read_json(parser.parse_args().config))
