"""Full-size random-input architecture checks, not training convergence or homework scores."""
import gc
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from common.utils import ROOT,configure,read_json,write_json
from quiz1.Code.models import RetrievalModel,supervised_contrastive
from quiz2.Code.models import FusionClassifier
from quiz3.Code.models import CaptionModel


def main():
    config=read_json(ROOT/"config.rtx5090.json");device=configure(config,"cuda");results=[]
    def probe(name,model,loss_function):
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        model=model.to(device).train();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        scaler=torch.amp.GradScaler("cuda",init_scale=1024)
        with torch.autocast("cuda",dtype=torch.float16):loss=loss_function(model)
        if not torch.isfinite(loss):raise RuntimeError(f"Nonfinite {name} loss")
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise RuntimeError("Nonfinite gradient")
        scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
        results.append({"model":name,"pass":True,"peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3})
        print(results[-1],flush=True)
        del model,optimizer,scaler,loss;gc.collect();torch.cuda.empty_cache()
    for kind in config["retrieval"]["backbones"]:
        batch=config["retrieval"]["classes_per_batch"]*config["retrieval"]["images_per_class"]
        images=torch.randn(batch,3,224,224,device=device)
        labels=torch.arange(config["retrieval"]["classes_per_batch"],device=device).repeat_interleave(config["retrieval"]["images_per_class"])
        def loss_fn(net):
            z,logits=net(images);return supervised_contrastive(z,labels)+.5*F.cross_entropy(logits,labels)
        probe("retrieval_"+kind,RetrievalModel(kind,200,pretrained=False),loss_fn)
        del images,labels
    for kind in config["multimodal"]["backbones"]:
        batch=config["multimodal"]["batch_size"]
        images=torch.randn(batch,3,224,224,device=device);text=torch.randn(batch,768,device=device);labels=torch.arange(batch,device=device)%200
        probe("fusion_"+kind,FusionClassifier(kind,200,pretrained=False),lambda net:F.cross_entropy(net(images,text),labels))
        del images,text,labels
    settings=config["caption"];batch=settings["batch_size"]
    features=torch.randn(batch,16,768,device=device);tokens=torch.randint(4,7380,(batch,settings["max_length"]),device=device)
    probe("clip_transformer_decoder",CaptionModel("transformer",7380,feature_dim=768),
          lambda net:F.cross_entropy(net(features,tokens[:,:-1]).reshape(-1,7380),tokens[:,1:].reshape(-1)))
    write_json(ROOT/"analysis/gpu_probe.json",{"scope":"random inputs, untrained weights; AMP backward/optimizer only",
        "gpu":torch.cuda.get_device_name(0),"results":results})


if __name__=="__main__":main()
