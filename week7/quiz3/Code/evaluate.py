"""Generate frozen before/after answers on held-out tests, using identical inference settings."""
import argparse
import importlib.metadata
import time

from common import GENERATION,MODELS,ROOT,VARIANTS,digest,read,require,save,sha,validate_data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind",choices=["llm","vlm"],required=True)
    parser.add_argument("--stage",choices=["before","after"],required=True)
    parser.add_argument("--limit",type=int,help="Smoke generation only, saved under Output/smoke_eval")
    args=parser.parse_args();validate_data()
    output=ROOT/("Output/smoke_eval" if args.limit else "Output")/args.kind/(args.stage+".json")
    require(not output.exists(),"Evaluation already exists; preserve it and use a separate experiment for intentional repeats")
    if args.kind=="llm":
        tasks=[dict(q,suite="target",variant="text") for q in read(ROOT/"Data/llm_test.json")]
        tasks += [dict(q,suite="general",variant="text") for q in read(ROOT/"Data/general_100.json")]
    else:
        tasks=[]
        for variant in VARIANTS:
            for q in read(ROOT/"Data/vlm_test.json"):
                relative=f"variants/{variant}/{q['id']}.png"
                tasks.append(dict(q,suite="visual",variant=variant,input_image=relative,input_image_sha256=sha(ROOT/"Data"/relative)))
    if args.limit:tasks=tasks[:args.limit]
    training_path=ROOT/f"Output/{args.kind}/training.json"
    adapter=None
    if args.stage=="after":
        training=read(training_path)
        require(training["complete"] is True and not training["smoke"],"Need a completed formal training run")
        require(training["data_manifest_sha256"]==sha(ROOT/"Data/manifest.json"),"Training data changed")
        adapter=ROOT/f"Output/{args.kind}/adapter"
        for name,h in training["adapter_sha256"].items():require(sha(adapter/name)==h,"Adapter changed")
    import torch
    from PIL import Image
    from transformers import GenerationConfig,set_seed
    from models import encode,load_model,processor_for
    set_seed(42);processor=processor_for(args.kind);model=load_model(args.kind,adapter=adapter).eval()
    native=model.generation_config
    config=GenerationConfig(**GENERATION,use_cache=True,eos_token_id=native.eos_token_id,bos_token_id=native.bos_token_id,
        pad_token_id=processor.pad_token_id if args.kind=="llm" else processor.tokenizer.pad_token_id)
    def generate(prompt,image=None):
        inputs=encode(args.kind,processor,prompt,image)
        inputs={k:v.to(device="cuda:0",dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
        torch.cuda.synchronize();tick=time.perf_counter()
        with torch.inference_mode():generated=model.generate(**inputs,generation_config=config)
        torch.cuda.synchronize();seconds=time.perf_counter()-tick
        new=generated[0,inputs['input_ids'].shape[-1]:];ids=new.cpu().tolist()
        tokenizer=processor if args.kind=="llm" else processor.tokenizer
        eos=config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
        return {"answer":tokenizer.decode(new,skip_special_tokens=True).strip(),"input_tokens":inputs['input_ids'].shape[-1],
            "output_tokens":len(ids),"seconds":seconds,"hit_token_limit":len(ids)>=config.max_new_tokens and (not ids or ids[-1] not in eos)}
    generate("Answer with one word: ready." if args.kind=="vlm" else "請只回答測試。",Image.new("RGB",(448,336),"white") if args.kind=="vlm" else None)
    identity=[{"id":q["id"],"suite":q["suite"],"variant":q["variant"],"prompt":q["prompt"],"image_sha256":q.get("input_image_sha256")} for q in tasks]
    meta={"schema_version":1,"kind":args.kind,"stage":args.stage,"smoke":bool(args.limit),"complete":False,
        "model":{k:v for k,v in MODELS[args.kind].items() if k!="cache"},"thinking":False if args.kind=="llm" else None,
        "generation_config":GENERATION,"data_manifest_sha256":sha(ROOT/"Data/manifest.json"),"request_digest":digest(identity),
        "gpu":torch.cuda.get_device_name(0),"packages":{name:importlib.metadata.version(name) for name in ["torch","transformers","peft","Pillow"]},
        "processor_config":processor.image_processor.to_dict() if args.kind=="vlm" else {"system_prompt":"請完成使用者任務，遵守指定的輸出格式。","enable_thinking":False},
        "native_token_ids":{"eos":config.eos_token_id,"bos":config.bos_token_id,"pad":config.pad_token_id},
        "training_sha256":sha(training_path) if adapter else None,"results":[]}
    save(output,meta)
    for index,(q,request) in enumerate(zip(tasks,identity),1):
        if args.kind=="vlm":
            with Image.open(ROOT/"Data"/q['input_image']) as im:answer=generate(q['prompt'],im.convert("RGB"))
        else:answer=generate(q['prompt'])
        meta["results"].append({**request,**answer})
        meta["complete"]=index==len(tasks);save(output,meta)
        print(f"{args.kind}/{args.stage} [{index}/{len(tasks)}] {q['id']} {q['variant']}: {answer['answer']}",flush=True)
    print("Saved "+str(output),flush=True)


if __name__=="__main__":main()
