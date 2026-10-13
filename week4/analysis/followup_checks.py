"""Quantify IMDb source duplicates and check decoding at the original batch size.

No training, original-output rewrites or Gemini calls.
"""
import collections
import hashlib
import sys
import tarfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from common.utils import ROOT, configure, read_json, write_json
from common.training import amp_context
from quiz4.Code.models import CaptionModel, decode, generate


def main():
    cfg = read_json(ROOT/"config.rtx5090.json")
    device = configure(cfg, "cuda")
    folder = ROOT/"quiz1"/"Data"/"imdb"
    data = {s: torch.load(folder/f"{s}.pt", weights_only=True) for s in ("train", "val", "test")}
    wanted = {sid for rows in data.values() for sid in rows["ids"]}
    hashes = {}
    # Sequential archive reads avoid opening 50,000 tiny Windows files again.
    with tarfile.open(ROOT/"quiz1"/"Data"/"downloads"/"aclImdb_v1.tar.gz", "r|gz") as archive:
        for member in archive:
            sid = member.name.removeprefix("aclImdb/")
            if member.isfile() and sid in wanted:
                hashes[sid] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    assert hashes.keys() == wanted
    print("IMDb archive contents verified", flush=True)
    train_hashes = {hashes[sid] for sid in data["train"]["ids"]}
    dev_hashes = train_hashes | {hashes[sid] for sid in data["val"]["ids"]}
    test_hashes = [hashes[sid] for sid in data["test"]["ids"]]
    masks = {"official": np.ones(len(test_hashes), dtype=bool),
             "exclude_train_exact_duplicates": np.array([h not in train_hashes for h in test_hashes]),
             "exclude_train_or_val_exact_duplicates": np.array([h not in dev_hashes for h in test_hashes])}
    result = {"imdb_duplicate_sensitivity": {}, "caption_original_batch_recheck": {}}
    for kind in ("rnn", "lstm"):
        with np.load(ROOT/"quiz2"/"Output"/kind/"probabilities.npz") as z:
            y, p = z["labels"], z["probabilities"]
            values = []
            for name, mask in masks.items():
                values.append({"subset": name, "samples": int(mask.sum()), "removed": int((~mask).sum()),
                               "accuracy": float(accuracy_score(y[mask], p[mask].argmax(1))),
                               "macro_f1": float(f1_score(y[mask], p[mask].argmax(1), average="macro")),
                               "positive_auc": float(roc_auc_score(y[mask], p[mask,1]))})
            result["imdb_duplicate_sensitivity"][kind] = values
    conflicts = collections.defaultdict(set)
    for split in data:
        for sid, label in zip(data[split]["ids"], data[split]["labels"].tolist()):
            conflicts[hashes[sid]].add(label)
    result["imdb_duplicate_label_conflicts_unique_hashes"] = sum(len(v) > 1 for v in conflicts.values())
    result["imdb_excluded_test_ids"] = {
        name: [sid for sid, include in zip(data["test"]["ids"], mask) if not include]
        for name, mask in masks.items() if name != "official"
    }
    s = cfg["caption"]
    vocab = read_json(ROOT/"quiz1"/"Data"/"flickr8k"/"vocab.json")
    model = CaptionModel("transformer", len(vocab), **{k:s[k] for k in
                         ("dim", "layers", "heads", "dropout", "max_length", "grid_size")})
    model.load_state_dict(torch.load(ROOT/"quiz4"/"Output"/"transformer"/"best.pt", weights_only=True)["model"])
    model.to(device).eval()
    features = torch.load(ROOT/"quiz4"/"Data"/"features"/"test.pt", weights_only=True)["features"]
    rows = read_json(ROOT/"quiz4"/"Output"/"transformer"/"beam"/"predictions.json")
    generated = []
    with torch.inference_mode(), amp_context(device):
        for start in range(0, len(features), s["decode_batch_size"]):
            batch = features[start:start+s["decode_batch_size"]].float().to(device)
            ids = generate(model, batch, s["max_length"], s["beam_width"], s["beam_length_penalty"])
            generated.extend(decode(row, vocab) for row in ids.cpu().tolist())
    mismatches = [{"index": i, "image_id": row["id"], "saved": row["caption"], "fresh": text}
                  for i, (row, text) in enumerate(zip(rows, generated)) if row["caption"] != text]
    result["caption_original_batch_recheck"] = {
        "model": "transformer", "decoding": "beam", "batch_size": s["decode_batch_size"],
        "exact_matches": len(rows)-len(mismatches), "total": len(rows), "mismatches": mismatches,
        "interpretation": "Small FP16 beam-ranking differences can depend on batch size/algorithm; compare at original batch size.",
    }
    write_json(ROOT/"analysis"/"followup_checks.json", result)
    print(result["imdb_duplicate_sensitivity"])
    print({k:v for k,v in result["caption_original_batch_recheck"].items() if k != "mismatches"})


if __name__ == "__main__":
    main()
