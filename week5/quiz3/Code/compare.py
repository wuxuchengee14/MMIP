"""Re-score saved Week4 and Week5 captions using exactly the same references/settings."""
from __future__ import annotations

import sys
import time
from pathlib import Path
if __package__ in (None, ""):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from common.utils import ROOT,digest,file_hash,read_json,write_csv,write_json
from quiz1.Code.prepare import week4_root
from quiz3.Code.bleu import corpus_bleu


def aligned_predictions(path,records):
    rows=read_json(path)
    if len({r["id"] for r in rows})!=len(rows):raise ValueError(f"Duplicate prediction IDs: {path}")
    mapping={r["id"]:r for r in rows}
    if set(mapping)!={r["id"] for r in records}:raise ValueError(f"Not the exact same full test set: {path}")
    if any(mapping[r["id"]].get("references")!=r["captions"] for r in records):
        raise ValueError(f"Different reference captions: {path}")
    if any(not isinstance(r.get("caption"),str) for r in rows):raise ValueError("Invalid caption type")
    return [mapping[r["id"]]["caption"] for r in records]


def run(config,device):
    from bert_score import BERTScorer
    original=week4_root(config)
    records=read_json(original/"quiz1/Data/flickr8k/test.json")
    references=[r["captions"] for r in records]
    settings=config["bertscore"]
    sources=[("week4",kind,mode,original/"quiz4/Output"/kind/mode)
             for kind in ("lstm","transformer") for mode in ("greedy","beam")]
    sources.extend(("week5","clip_transformer",mode,ROOT/"quiz3/Output/clip_transformer"/mode) for mode in ("greedy","beam"))
    prepared=[]
    for week,kind,mode,folder in sources:
        path=folder/"predictions.json"
        captions=aligned_predictions(path,records)
        identity=digest({"predictions":file_hash(path),"references":references,"settings":settings,"version":1})
        cache=ROOT/"quiz3/Output/bertscore"/f"{week}_{kind}_{mode}"
        prepared.append((week,kind,mode,folder,captions,identity,cache))
    scorer=None;comparison=[];revisions=set()
    for week,kind,mode,folder,captions,identity,cache in prepared:
        marker=cache/"metrics.json"
        if marker.exists():
            metrics=read_json(marker)
            if metrics["identity"]!=identity:raise ValueError("BERTScore inputs/settings changed; use a fresh score cache")
        else:
            if scorer is None:
                scorer=BERTScorer(model_type=settings["model_type"],num_layers=settings["num_layers"],
                    batch_size=settings["batch_size"],device=str(device),idf=settings["idf"],
                    rescale_with_baseline=settings["rescale_with_baseline"],use_fast_tokenizer=settings["use_fast_tokenizer"])
            print(f"BERTScore {week}/{kind}/{mode}: {len(captions)} images, five references each",flush=True)
            # bert-score 0.3.13 shadows its timer variable in multi-reference mode.
            # Measure our own elapsed time and suppress its incorrect verbose duration.
            tick=time.perf_counter()
            p,r,f=scorer.score(captions,references,batch_size=settings["batch_size"],verbose=False)
            scoring_seconds=time.perf_counter()-tick
            print(f"BERTScore evaluation finished in {scoring_seconds:.2f}s",flush=True)
            # Library multi-reference mode takes the maximum separately for P, R, and F1.
            metrics={"identity":identity,"precision":float(p.mean()),"recall":float(r.mean()),"f1":float(f.mean()),
                "hash":scorer.hash,"settings":settings,"samples":len(records),
                "multi_reference":"library maximum over five references separately for P/R/F1",
                "model_revision":getattr(scorer._model.config,"_commit_hash",None),"scoring_seconds":scoring_seconds}
            cache.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(cache/"per_image.npz",ids=[r["id"] for r in records],
                                precision=p.cpu().numpy(),recall=r.cpu().numpy(),f1=f.cpu().numpy())
            write_json(marker,metrics)
        revisions.add(metrics.get("model_revision"))
        bleu=corpus_bleu(references,captions)
        timing=read_json(folder/"timing.json")
        comparison.append({"week":week,"model":kind,"decoding":mode,**{k:bleu[k] for k in ("bleu1","bleu2","bleu3","bleu4")},
                           "bertscore_precision":metrics["precision"],"bertscore_recall":metrics["recall"],"bertscore_f1":metrics["f1"],
                           "bertscore_hash":metrics["hash"],"bertscore_model_revision":metrics.get("model_revision"),
                           "decoder_mean_ms":timing["decoder_mean_ms"],
                           "estimated_pipeline_mean_ms":timing["estimated_pipeline_mean_ms"],"test_images":len(records),
                           "empty_captions":sum(not c.strip() for c in captions)})
    if len({r["bertscore_hash"] for r in comparison})!=1:
        raise ValueError("Different BERTScore evaluator hashes; recompute the six score caches with one environment")
    if len(revisions)!=1:
        raise ValueError("Different BERTScore model revisions; do not mix evaluator checkpoints")
    write_json(ROOT/"quiz3/Output/comparison.json",comparison);write_csv(ROOT/"quiz3/Output/comparison.csv",comparison)
    if scorer is not None:del scorer
    if device.type=="cuda":torch.cuda.empty_cache()
    return comparison
