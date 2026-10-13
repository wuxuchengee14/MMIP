from __future__ import annotations

import collections
import hashlib
import html
import json
import re
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from common.utils import ROOT, digest, file_hash, read_json, report, table, write_csv, write_json

DATA = ROOT / "quiz1" / "Data"
IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
FLICKR_BASE = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/"
SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]


def tokenize(text):
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[.!?,;:]", text.lower())


def vocabulary(texts, size):
    counts = collections.Counter(t for text in texts for t in tokenize(text))
    # Tie-breaking is stable and independent of operating-system file ordering.
    words = sorted(counts, key=lambda w: (-counts[w], w))
    return SPECIAL + [w for w in words if w not in SPECIAL][:size - len(SPECIAL)]


def encode(text, words, length, caption=False):
    ids = [words.get(t, 1) for t in tokenize(text)]
    ids = ([2] + ids[:length - 2] + [3]) if caption else (ids[:length] or [1])
    return ids


def download(url, destination, md5=None):
    from common.network import enable_system_trust
    enable_system_trust()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        tmp = destination.with_suffix(destination.suffix + ".part")
        print(f"Download {url}", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "MMIP-week4/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as f:
            total = int(response.headers.get("Content-Length", 0))
            done, last = 0, 0
            while block := response.read(1024 * 1024):
                f.write(block)
                done += len(block)
                if done - last >= 20 * 1024 * 1024:
                    print(f"  {done / 1024**2:.0f} / {total / 1024**2:.0f} MiB", flush=True)
                    last = done
        if total and done != total:
            raise IOError("Incomplete download; retry the same command.")
        tmp.replace(destination)
    if md5:
        h = hashlib.md5()
        with destination.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != md5:
            raise ValueError(f"Checksum mismatch: {destination}; remove the corrupt archive and retry.")
    return destination


def safe_destination(root, name):
    root = Path(root).resolve()
    candidate = (root / name).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Unsafe archive member: {name}")
    return candidate


def extract_zip(archive, root):
    marker = Path(root) / (Path(archive).name + ".extracted")
    if marker.exists():
        return
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            destination = safe_destination(root, item.filename)
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with z.open(item) as src, destination.open("wb") as out:
                    shutil.copyfileobj(src, out)
    marker.write_text("ok", encoding="utf-8")


def prepare_imdb(config):
    folder = DATA / "imdb"
    signature = digest({"seed": config["seed"], "max_length": config["text"]["max_length"],
                        "vocab_size": config["text"]["vocab_size"], "version": 1})
    manifest_file = folder / "manifest.json"
    if manifest_file.exists():
        manifest = read_json(manifest_file)
        if manifest["signature"] != signature:
            raise RuntimeError("IMDb preprocessing config changed. Use the original config or a new data directory.")
        if file_hash(folder / "vocab.json") != manifest["vocab_sha256"]:
            raise ValueError("IMDb vocabulary was modified.")
        for split in ("train", "val", "test"):
            if file_hash(folder / f"{split}.pt") != manifest["hashes"][f"{split}.pt"]:
                raise ValueError(f"IMDb prepared {split} data was modified.")
        return manifest
    archive = download(IMDB_URL, DATA / "downloads" / "aclImdb_v1.tar.gz",
                       "7c2ac02c03563afcf9b574c7e56c153a")
    raw = folder / "raw"
    marker = raw / ".extracted"
    if not marker.exists():
        with tarfile.open(archive) as tar:
            for member in tar:
                if member.isfile() and re.match(r"aclImdb/(train|test)/(pos|neg)/[^/]+\.txt$", member.name):
                    destination = safe_destination(raw, member.name)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src, destination.open("wb") as out:
                        shutil.copyfileobj(src, out)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok", encoding="utf-8")
    records = {}
    for split in ("train", "test"):
        records[split] = [
            {"id": f"{split}/{label}/{p.name}", "label": y, "text": p.read_text(encoding="utf-8")}
            for label, y in (("neg", 0), ("pos", 1))
            for p in sorted((raw / "aclImdb" / split / label).glob("*.txt"))
        ]
        if len(records[split]) != 25000:
            raise ValueError(f"Unexpected IMDb {split} size: {len(records[split])}")
    train, val = train_test_split(records["train"], test_size=0.2, random_state=config["seed"],
                                 stratify=[r["label"] for r in records["train"]])
    splits = {"train": train, "val": val, "test": records["test"]}
    vocab = vocabulary((r["text"] for r in train), config["text"]["vocab_size"])
    word_ids = {word: i for i, word in enumerate(vocab)}
    write_json(folder / "vocab.json", vocab)
    write_csv(folder / "split_ids.csv", [
        {"id": r["id"], "split": split, "label": r["label"]} for split, rows in splits.items() for r in rows
    ])
    for split, rows in splits.items():
        seqs = [encode(r["text"], word_ids, config["text"]["max_length"]) for r in rows]
        tokens = torch.zeros(len(rows), config["text"]["max_length"], dtype=torch.long)
        for i, ids in enumerate(seqs):
            tokens[i, :len(ids)] = torch.tensor(ids)
        torch.save({"tokens": tokens, "lengths": torch.tensor([len(s) for s in seqs]),
                    "labels": torch.tensor([r["label"] for r in rows]), "ids": [r["id"] for r in rows]},
                   folder / f"{split}.pt")
    manifest = {"source": IMDB_URL, "signature": signature, "classes": ["negative", "positive"],
                "counts": {s: len(r) for s, r in splits.items()}, "vocab_size": len(vocab),
                "vocab_sha256": file_hash(folder / "vocab.json"),
                "hashes": {f"{s}.pt": file_hash(folder / f"{s}.pt") for s in splits},
                "examples": [{"text": r["text"][:500], "label": r["label"]} for r in train[:4]]}
    write_json(manifest_file, manifest)
    return manifest


def prepare_organ(config):
    from medmnist import INFO
    folder = DATA / "organamnist"
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / "organamnist.npz"
    old = ROOT.parent / "week3_homework" / "data" / "organamnist.npz"
    if not destination.exists() and old.exists():
        shutil.copy2(old, destination)
    info = INFO["organamnist"]
    download(info["url"], destination, info["MD5"])
    with np.load(destination) as d:
        counts = {split: len(d[f"{split}_labels"]) for split in ("train", "val", "test")}
        classes = [info["label"][str(i)] for i in range(11)]
        balance = [{"split": s, "class": classes[i], "count": int((d[f"{s}_labels"] == i).sum())}
                   for s in counts for i in range(11)]
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 11, figsize=(16, 3))
        for i, ax in enumerate(axes):
            idx = np.flatnonzero(d["train_labels"].ravel() == i)[0]
            ax.imshow(d["train_images"][idx], cmap="gray")
            ax.set_title(classes[i], fontsize=7)
            ax.axis("off")
        fig.tight_layout()
        output = ROOT / "quiz1" / "Output"
        output.mkdir(parents=True, exist_ok=True)
        fig.savefig(output / "organ_examples.png", dpi=150)
        plt.close(fig)
    manifest = {"source": info["url"], "sha256": file_hash(destination), "counts": counts,
                "classes": classes, "split": "Official train/val/test; same for ViT and ResNet",
                "limitation": "28x28 resized to 224 adds no spatial detail; no patient IDs available here."}
    write_csv(ROOT / "quiz1" / "Output" / "organ_class_counts.csv", balance)
    write_json(folder / "manifest.json", manifest)
    return manifest


def prepare_flickr(config):
    folder = DATA / "flickr8k"
    for filename in ("Flickr8k_Dataset.zip", "Flickr8k_text.zip"):
        archive = download(FLICKR_BASE + filename, DATA / "downloads" / filename)
        extract_zip(archive, folder)
    def find(name):
        matches = sorted(folder.rglob(name))
        if len(matches) != 1:
            raise ValueError(f"Expected one {name}, got {len(matches)}")
        return matches[0]
    captions = collections.defaultdict(list)
    for line in find("Flickr8k.token.txt").read_text(encoding="utf-8").splitlines():
        name, text = line.split("\t", 1)
        captions[name.rsplit("#", 1)[0]].append(text)
    images = {p.name: p.relative_to(folder).as_posix() for p in folder.rglob("*.jpg")}
    filenames = {"train": "Flickr_8k.trainImages.txt", "val": "Flickr_8k.devImages.txt",
                 "test": "Flickr_8k.testImages.txt"}
    splits = {}
    seen = set()
    for split, filename in filenames.items():
        names = [n.strip() for n in find(filename).read_text().splitlines() if n.strip()]
        if seen.intersection(names) or len(names) != len(set(names)):
            raise ValueError("Overlapping Flickr image split")
        seen.update(names)
        splits[split] = []
        for name in names:
            if name not in images or len(captions[name]) != 5:
                raise ValueError(f"Missing image or references: {name}")
            splits[split].append({"id": name, "image": images[name], "captions": captions[name]})
    expected = {"train": 6000, "val": 1000, "test": 1000}
    if {s: len(r) for s, r in splits.items()} != expected:
        raise ValueError("Unexpected Flickr official split sizes")
    vocab = vocabulary((c for r in splits["train"] for c in r["captions"]), config["caption"]["vocab_size"])
    write_json(folder / "vocab.json", vocab)
    for split, rows in splits.items():
        write_json(folder / f"{split}.json", rows)
    manifest = {
        "source": FLICKR_BASE, "counts": expected, "references_per_image": 5,
        "vocab_size": len(vocab), "vocab_sha256": file_hash(folder / "vocab.json"),
        "max_length": config["caption"]["max_length"],
        "hashes": {f"{s}.json": file_hash(folder / f"{s}.json") for s in splits},
        "archives": {n: file_hash(DATA / "downloads" / n) for n in ("Flickr8k_Dataset.zip", "Flickr8k_text.zip")},
    }
    write_json(folder / "manifest.json", manifest)
    return manifest


def run(config):
    DATA.mkdir(parents=True, exist_ok=True)
    imdb, organ, flickr = prepare_imdb(config), prepare_organ(config), prepare_flickr(config)
    manifests = {"imdb": imdb, "organamnist": organ, "flickr8k": flickr}
    write_json(ROOT / "quiz1" / "Output" / "datasets.json", manifests)
    body = "三種資料均使用固定、互斥的訓練／驗證／測試集合。\n\n"
    body += table([{"dataset": n, **m["counts"]} for n, m in manifests.items()])
    body += (
        "\n\n## 資料來源與處理\n\n"
        f"- IMDb: {IMDB_URL}；官方訓練集分層切出 20% validation；測試集保持不變。"
        "移除 HTML、轉小寫、以正規表示式斷詞，詞彙表只由 training 建立。\n"
        "- OrganAMNIST: https://medmnist.com/v2；11 類器官 CT，原始 28×28 灰階。"
        "官方 train/val/test 分割；複製既有原始檔並檢查官方 MD5，不沿用第三週模型。\n"
        "- Flickr8k: 使用 TensorFlow 官方教學所引用的下載鏡像；依官方影像清單分割，"
        "每張圖的五段描述不跨集合。詞彙表只由 training captions 建立。\n"
        "- Flickr8k 鏡像來源與說明：https://www.tensorflow.org/text/tutorials/image_captioning\n"
        "\n## 限制\n\nOrganAMNIST 放大不會增加細節；這裡沒有病患 ID，無法另行驗證病患層級重疊。"
        "測試集只用於最終評估，不用來挑選權重或建立詞彙表。\n\n"
        "類別分布見 organ_class_counts.csv，器官範例見 organ_examples.png。"
        "文字範例見 datasets.json，圖片與五段描述可由 Data/flickr8k/train.json 查閱。"
    )
    report("quiz1", "Quiz 1：資料集準備", body)
    return manifests


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.rtx5090.json")
    run(read_json(parser.parse_args().config))
