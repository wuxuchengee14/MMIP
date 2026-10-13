"""Actual RTX 5090 batch-memory probe; random data, no pretrained downloads.

Uses real configured architectures and batch sizes with AMP + backward + AdamW.
This checks compatibility/memory, not convergence or complete training duration.
"""
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from common.utils import ROOT, configure, read_json, synchronize, write_json
from quiz2.Code.models import SentimentModel
from quiz3.Code.train import build_model, preprocess
from quiz4.Code.models import CaptionModel


def main():
    config = read_json(ROOT/"config.rtx5090.json")
    device = configure(config, "cuda")
    rows = []
    for task in ("rnn", "lstm", "vit_b_16", "resnet50", "caption_lstm", "caption_transformer"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        if task in ("rnn", "lstm"):
            s = config["text"]
            model = SentimentModel(task, s["vocab_size"], s["embedding_dim"], s["hidden_dim"], s["dropout"]).to(device)
            args = (torch.randint(1, s["vocab_size"], (s["batch_size"], s["max_length"]), device=device),
                    torch.full((s["batch_size"],), s["max_length"], device=device))
            labels = torch.randint(0, 2, (s["batch_size"],), device=device)
        elif task in ("vit_b_16", "resnet50"):
            s = config["vision"]
            model, _ = build_model(task, pretrained=False)
            model.to(device)
            args = (preprocess(torch.randint(0, 256, (s["batch_size"], 1, 28, 28),
                                            dtype=torch.uint8), device),)
            labels = torch.randint(0, 11, (s["batch_size"],), device=device)
        else:
            s = config["caption"]
            model = CaptionModel(task.removeprefix("caption_"), s["vocab_size"], **{k: s[k] for k in
                                 ("dim", "layers", "heads", "dropout", "max_length", "grid_size")}).to(device)
            args = (torch.randn(s["batch_size"], s["grid_size"]**2, 2048, device=device),
                    torch.randint(1, s["vocab_size"], (s["batch_size"], s["max_length"]-1), device=device))
            labels = torch.randint(1, s["vocab_size"], (s["batch_size"], s["max_length"]-1), device=device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(*args)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        synchronize(device)
        assert torch.isfinite(loss)
        row = {"model": task, "batch_size": s["batch_size"],
               "peak_allocated_gib": torch.cuda.max_memory_allocated()/1024**3,
               "peak_reserved_gib": torch.cuda.max_memory_reserved()/1024**3,
               "single_step_with_initialization_seconds": time.perf_counter()-tick}
        rows.append(row)
        print(row, flush=True)
        del model, opt, scaler, args, labels, logits, loss
        gc.collect()
    write_json(ROOT/"tests"/"gpu_probe_results.json", {
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "results": rows,
        "scope": "Random inputs, real six model architectures/configured batch sizes; one step each; not full training."})


if __name__ == "__main__":
    main()
