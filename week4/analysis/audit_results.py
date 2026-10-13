"""Independent read-only review of saved Week 4 artifacts; no training/API.

Only writes analysis/results_audit.json. Original results remain unchanged.
"""
from __future__ import annotations
import collections
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from common.utils import ROOT, configure, file_hash, read_json, write_json
from common.training import amp_context
from quiz1.Code.prepare import tokenize
from quiz2.Code.models import SentimentModel
from quiz3.Code.train import OrganDataset, build_model, preprocess
from quiz4.Code.models import CaptionModel, decode, generate


def history(path):
    with path.open(encoding="utf-8-sig") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def metrics(y, p):
    nclasses = p.shape[1]
    normalized = p.astype(np.float64)
    normalized /= normalized.sum(axis=1, keepdims=True)
    return {
        "n": len(y), "top1": float(accuracy_score(y, p.argmax(1))),
        "top5": float(np.any(np.argsort(p, axis=1)[:, -min(5, nclasses):] == y[:, None], axis=1).mean()),
        "macro_f1": float(f1_score(y, p.argmax(1), average="macro")),
        "macro_auc": float(np.mean([roc_auc_score(y == i, p[:, i]) for i in range(nclasses)])),
        "nll": float(log_loss(y, normalized, labels=np.arange(nclasses))),
        "mean_confidence": float(p.max(1).mean()),
        "wrong_mean_confidence": float(p.max(1)[p.argmax(1) != y].mean()),
        "correct": int((y == p.argmax(1)).sum()), "errors": int((y != p.argmax(1)).sum()),
    }


def overlaps(splits):
    result = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        left, right = collections.Counter(splits[a]), collections.Counter(splits[b])
        intersection = left.keys() & right.keys()
        result[f"{a}_{b}"] = {
            "shared_unique_hashes": len(intersection),
            "left_rows_in_overlap": sum(left[h] for h in intersection),
            "right_rows_in_overlap": sum(right[h] for h in intersection),
        }
    return result


def independent_bleu(rows):
    # Independent implementation of max-reference clipped n-gram precision.
    counts = np.zeros(4, dtype=np.int64)
    matched = np.zeros(4, dtype=np.int64)
    hyp_len = ref_len = 0
    for row in rows:
        h = tokenize(row["caption"])
        refs = [tokenize(r) for r in row["references"]]
        hyp_len += len(h)
        ref_len += sorted((len(r) for r in refs), key=lambda n: (abs(n-len(h)), n))[0]
        for n in range(1, 5):
            hgrams = collections.Counter(zip(*(h[k:] for k in range(n))))
            counts[n-1] += sum(hgrams.values())
            rgrams = [collections.Counter(zip(*(r[k:] for k in range(n)))) for r in refs]
            matched[n-1] += sum(min(c, max(r.get(g, 0) for r in rgrams)) for g, c in hgrams.items())
    p = matched / np.maximum(1, counts)
    bp = math.exp(min(0, 1-ref_len/hyp_len)) if hyp_len else 0
    return {f"bleu{n}": float(bp * np.prod(p[:n])**(1/n)) for n in range(1, 5)}


def main():
    cfg = read_json(ROOT/"config.rtx5090.json")
    device = configure(cfg, "cuda")
    result = {"classification": {}, "caption": {}, "splits": {}}
    imdb_folder = ROOT/"quiz1"/"Data"/"imdb"
    imdb = {s: torch.load(imdb_folder/f"{s}.pt", weights_only=True) for s in ("train", "val", "test")}
    ids = {s: imdb[s]["ids"] for s in imdb}
    result["splits"]["imdb_ids"] = overlaps(ids)
    raw_hashes = {}
    for split in imdb:
        raw_hashes[split] = [file_hash(imdb_folder/"raw"/"aclImdb"/sid) for sid in imdb[split]["ids"]]
    result["splits"]["imdb_raw_text"] = overlaps(raw_hashes)
    organ_file = ROOT/"quiz1"/"Data"/"organamnist"/"organamnist.npz"
    with np.load(organ_file) as d:
        organ_labels = d["test_labels"].ravel().copy()
        result["splits"]["organ_exact_pixels"] = overlaps({
            s: [hashlib.sha256(im.tobytes()).hexdigest() for im in d[f"{s}_images"]]
            for s in ("train", "val", "test")
        })
    for quiz, kinds in (("quiz2", ("rnn", "lstm")), ("quiz3", ("vit_b_16", "resnet50"))):
        for kind in kinds:
            directory = ROOT/quiz/"Output"/kind
            done = read_json(directory/"complete.json")
            h = history(directory/"history.csv")
            with np.load(directory/"probabilities.npz") as saved:
                y, probs = saved["labels"], saved["probabilities"]
                expected_y = imdb["test"]["labels"].numpy() if quiz == "quiz2" else organ_labels
                assert np.array_equal(y, expected_y), f"Wrong test labels {kind}"
                recomputed = metrics(y, probs)
                recorded = read_json(directory/"metrics.json")
                delta = {k: abs(recomputed[k] - recorded[k]) for k in ("top1", "top5", "macro_f1", "macro_auc")}
                assert max(delta.values()) < 1e-10
                pair = {"metrics": recomputed, "metric_deltas": delta, "best_epoch": done["best_epoch"],
                        "best_epoch_history": h[done["best_epoch"]-1], "last_epoch_history": h[-1]}
                checkpoint = torch.load(directory/"best.pt", weights_only=True, map_location="cpu")
                assert checkpoint["epoch"] == done["best_epoch"]
                if quiz == "quiz2":
                    s = cfg["text"]
                    model = SentimentModel(kind, len(read_json(imdb_folder/"vocab.json")),
                                           s["embedding_dim"], s["hidden_dim"], s["dropout"])
                else:
                    model, _ = build_model(kind, pretrained=False)
                model.load_state_dict(checkpoint["model"])
                model.to(device).eval()
                predictions = {}
                for split in ("val", "test"):
                    if quiz == "quiz2":
                        data = imdb[split]
                        ds = TensorDataset(data["tokens"], data["lengths"], data["labels"])
                    else:
                        ds = OrganDataset(organ_file, split)
                    batches = DataLoader(ds, batch_size=128, num_workers=0)
                    p = []
                    with torch.inference_mode(), amp_context(device):
                        for batch in batches:
                            if quiz == "quiz2":
                                logits = model(batch[0].to(device), batch[1].to(device))
                            else:
                                logits = model(preprocess(batch[0], device))
                            p.append(logits.float().softmax(-1).cpu().numpy())
                    probabilities = np.concatenate(p)
                    labels = imdb[split]["labels"].numpy() if quiz == "quiz2" else ds.labels
                    predictions[split] = metrics(labels, probabilities)
                    if split == "test":
                        pair["fresh_test_vs_saved_max_prob_delta"] = float(np.max(np.abs(probabilities-probs)))
                        pair["fresh_test_vs_saved_disagreements"] = int((probabilities.argmax(1) != probs.argmax(1)).sum())
                pair["fresh_checkpoint_evaluation"] = predictions
                result["classification"][kind] = pair
                if quiz == "quiz3":
                    pairs = collections.Counter(zip(y[y != probs.argmax(1)].tolist(),
                                                     probs.argmax(1)[y != probs.argmax(1)].tolist()))
                    names = read_json(ROOT/"quiz1"/"Data"/"organamnist"/"manifest.json")["classes"]
                    pair["top_confusions"] = [{"true": names[a], "predicted": names[b], "count": n}
                                             for (a, b), n in pairs.most_common(6)]
                del model, checkpoint
                torch.cuda.empty_cache()
                print(f"Classification verified: {kind}", flush=True)
    folder = ROOT/"quiz1"/"Data"/"flickr8k"
    records = {s: read_json(folder/f"{s}.json") for s in ("train", "val", "test")}
    result["splits"]["flickr_ids"] = overlaps({s: [r["id"] for r in rows] for s, rows in records.items()})
    result["splits"]["flickr_exact_jpeg_bytes"] = overlaps({
        s: [file_hash(folder/r["image"]) for r in rows] for s, rows in records.items()
    })
    test_by_id = {r["id"]: r for r in records["test"]}
    test_features = torch.load(ROOT/"quiz4"/"Data"/"features"/"test.pt", weights_only=True)
    assert test_features["ids"] == [r["id"] for r in records["test"]]
    vocab = read_json(folder/"vocab.json")
    feature_info = read_json(ROOT/"quiz4"/"Data"/"features"/"complete.json")
    for kind in ("lstm", "transformer"):
        directory = ROOT/"quiz4"/"Output"/kind
        done = read_json(directory/"complete.json")
        h = history(directory/"history.csv")
        s = cfg["caption"]
        model = CaptionModel(kind, len(vocab), **{k:s[k] for k in
                             ("dim", "layers", "heads", "dropout", "max_length", "grid_size")})
        checkpoint = torch.load(directory/"best.pt", weights_only=True)
        assert checkpoint["epoch"] == done["best_epoch"]
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        audit = {"best_epoch": done["best_epoch"], "epochs_ran": done["epochs_ran"],
                 "best_epoch_history": h[done["best_epoch"]-1], "last_epoch_history": h[-1],
                 "shared_feature_identity_matches": done["identity"]["features"] == feature_info["identity"],
                 "decoding": {}}
        for mode, width in (("greedy", 1), ("beam", s["beam_width"])):
            rows = read_json(directory/mode/"predictions.json")
            assert len(rows) == 1000 and len({r["id"] for r in rows}) == 1000
            assert set(test_by_id) == {r["id"] for r in rows}
            assert all(r["references"] == test_by_id[r["id"]]["captions"] for r in rows)
            scores = independent_bleu(rows)
            saved = read_json(directory/mode/"bleu.json")
            deltas = {k: abs(v-saved[k]) for k,v in scores.items()}
            assert max(deltas.values()) < 1e-12
            # Fresh inference on a fixed, spread-out subset; no reference tokens used.
            indices = np.linspace(0, 999, 20, dtype=int)
            fresh = []
            with torch.inference_mode(), amp_context(device):
                for index in indices:
                    ids = generate(model, test_features["features"][index:index+1].float().to(device),
                                   s["max_length"], width, s["beam_length_penalty"])
                    fresh.append(decode(ids[0].tolist(), vocab))
            matches = sum(text == rows[index]["caption"] for index, text in zip(indices, fresh))
            captions = [r["caption"] for r in rows]
            frequencies = collections.Counter(captions)
            audit["decoding"][mode] = {
                "bleu": scores, "bleu_deltas": deltas, "test_images": len(rows),
                "fresh_batch1_exact_matches_of_20": matches,
                "empty_captions": sum(not c.strip() for c in captions),
                "unknown_token_captions": sum("<unk>" in c for c in captions),
                "unique_captions": len(frequencies), "most_repeated": frequencies.most_common(4),
                "mean_words": float(np.mean([len(tokenize(c)) for c in captions])),
                "all_references_and_image_ids_match_test": True,
            }
        result["caption"][kind] = audit
        del model, checkpoint
        torch.cuda.empty_cache()
        print(f"Caption verified: {kind}", flush=True)
    result["gemini"] = read_json(ROOT/"quiz4"/"Output"/"gemini"/"summary.json")
    result["gemini_errors"] = read_json(ROOT/"quiz4"/"Output"/"gemini"/"errors.json")
    write_json(ROOT/"analysis"/"results_audit.json", result)
    print(json.dumps(result["splits"], indent=2))


if __name__ == "__main__":
    main()
