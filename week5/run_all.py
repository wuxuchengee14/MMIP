"""Sequential Week5 workflow; full completion requires actual outputs for all tasks."""
import argparse
import time
from pathlib import Path
from common.utils import ROOT,configure,read_json
from common.reports import completion,make_reports


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=ROOT/"config.rtx5090.json")
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    parser.add_argument("--stage",choices=("all","prepare","retrieval","multimodal","caption","compare","report"),default="all")
    args=parser.parse_args();config=read_json(args.config)
    tick=time.perf_counter()
    try:
        if args.stage in ("all","prepare"):
            from quiz1.Code.prepare import run
            print("[1/5] Prepare CUB images/text + read-only Week4 Flickr",flush=True);run(config)
        if args.stage in ("all","retrieval"):
            from quiz1.Code.prepare import prepare_cub
            from quiz1.Code.train import run
            prepare_cub(config)
            print("[2/5] Train CNN and ViT retrieval",flush=True);run(config,configure(config,args.device))
        if args.stage in ("all","multimodal"):
            from quiz1.Code.prepare import prepare_cub
            from quiz2.Code.train import run
            prepare_cub(config)
            print("[3/5] Train two image/text fusion classifiers",flush=True);run(config,configure(config,args.device))
        if args.stage in ("all","caption"):
            from quiz1.Code.prepare import prepare_flickr
            from quiz3.Code.train import run
            prepare_flickr(config)
            print("[4/5] Frozen CLIP + trainable Transformer captioning",flush=True);run(config,configure(config,args.device))
        if args.stage in ("all","compare"):
            from quiz1.Code.prepare import prepare_flickr
            from quiz3.Code.compare import run
            prepare_flickr(config)
            print("[5/5] Uniform BLEU/BERTScore comparison with Week4",flush=True);run(config,configure(config,args.device))
    finally:
        make_reports(config)
        done=completion(config)
    print(f"Stage {args.stage} finished in {time.perf_counter()-tick:.1f}s; full assignment complete={done}",flush=True)
    return 3 if args.stage in ("all","report","compare") and not done else 0


if __name__=="__main__":raise SystemExit(main())
