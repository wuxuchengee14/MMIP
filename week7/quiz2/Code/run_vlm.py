"""Run one pinned open VLM on the same 20 images/prompts; BF16, batch 1, CUDA only."""
import argparse
import importlib.metadata
import time
from datetime import datetime, timezone
from pathlib import Path

from common import GENERATION, MODELS, ROOT, digest, image_path, load_benchmark, read, request_identity, require, save


def load_processor(key, revision):
    from transformers import AutoProcessor
    options = {"revision":revision, "cache_dir":str(ROOT/"models"), "use_fast":False,
               "trust_remote_code":False}
    # Limit Qwen's dynamic resize to <=512 vision tokens (before any rounding).
    # SmolVLM uses its own published preprocessing, including image splitting.
    if key == "qwen":
        options.update(min_pixels=64*28*28, max_pixels=512*28*28)
    return AutoProcessor.from_pretrained(MODELS[key],**options)


def encode(processor, image, prompt):
    # Only the image and the user prompt enter the model. Reference/rubric/ID never do.
    messages = [{"role":"user", "content":[{"type":"image"}, {"type":"text", "text":prompt}]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], images=[image], return_tensors="pt")
    return inputs, rendered


def to_cuda(inputs):
    import torch
    # BatchFeature.to(device) alone leaves floating image tensors in FP32 for some processors.
    return {k:v.to(device="cuda:0", dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
            if isinstance(v, torch.Tensor) else v for k,v in inputs.items()}


def generate_answer(model, processor, image, prompt, generation):
    import torch
    inputs, rendered = encode(processor,image,prompt)
    inputs = to_cuda(inputs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs,generation_config=generation)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    new = output[0,inputs["input_ids"].shape[-1]:]
    ids = new.cpu().tolist()
    eos = generation.eos_token_id
    eos_ids = eos if isinstance(eos,list) else [eos]
    return {"answer":processor.batch_decode(new.unsqueeze(0),skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip(),
            "input_tokens":int(inputs["input_ids"].shape[-1]), "output_tokens":len(ids),
            "generation_seconds":seconds, "hit_token_limit":len(ids)>=generation.max_new_tokens and (not ids or ids[-1] not in eos_ids),
            "peak_torch_allocated_bytes":torch.cuda.max_memory_allocated(0),
            "rendered_chat":rendered}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--benchmark",type=Path,default=ROOT/"Data/benchmark.json")
    parser.add_argument("--runs-dir",type=Path,default=ROOT/"Output/full")
    parser.add_argument("--overwrite",action="store_true")
    args = parser.parse_args()
    questions = load_benchmark(args.benchmark)
    output = args.runs_dir/(args.model+".json")
    require(not output.exists() or args.overwrite, "Output exists; use another --runs-dir or explicitly --overwrite")
    lock = read(ROOT/"Data/models.lock.json")
    item = lock["models"][args.model]
    require(item["model_id"] == MODELS[args.model], "Model lock mismatch")
    import truststore
    truststore.inject_into_ssl()
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, GenerationConfig, set_seed
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),"CUDA BF16 required; run setup_vlm.py")
    set_seed(42)
    processor = load_processor(args.model,item["revision"])
    print(f"Loading {item['model_id']} in BF16. The first run downloads weights.",flush=True)
    model = AutoModelForImageTextToText.from_pretrained(item["model_id"], revision=item["revision"],
        cache_dir=str(ROOT/"models"), torch_dtype=torch.bfloat16, device_map={"":0},
        low_cpu_mem_usage=True, attn_implementation="sdpa", trust_remote_code=False).eval()
    require(getattr(model.config,"_commit_hash",None) == item["revision"],"Loaded model revision differs from lock")
    require(all(p.device.type == "cuda" for p in model.parameters()),"All parameters must be on GPU; CPU offload is disabled")
    native = model.generation_config
    generation = GenerationConfig(**GENERATION, repetition_penalty=1.0, length_penalty=1.0,
        eos_token_id=native.eos_token_id, bos_token_id=native.bos_token_id,
        pad_token_id=native.pad_token_id if native.pad_token_id is not None else processor.tokenizer.pad_token_id)
    require(generation.eos_token_id is not None,"Missing native EOS token")
    # Identical non-benchmark warmup image/prompt, excluded from the 20 scored rows.
    generate_answer(model,processor,Image.new("RGB",(640,480),"white"),"What is the background color? Answer in one word.",generation)
    package_names = ["torch","torchvision","transformers","accelerate","Pillow","num2words","huggingface-hub"]
    meta = {"schema_version":1, "model_key":args.model, **item, "precision":"bf16",
        "attention":"sdpa", "batch_size":1, "seed":42, "generation_config":GENERATION,
        "repetition_penalty":1.0, "length_penalty":1.0,
        "native_token_ids":{"eos":generation.eos_token_id,"bos":generation.bos_token_id,"pad":generation.pad_token_id},
        "benchmark_digest":digest(questions), "request_digest":digest([request_identity(q) for q in questions]),
        "gpu":torch.cuda.get_device_name(0), "cuda_runtime":torch.version.cuda,
        "parameter_count":sum(p.numel() for p in model.parameters()),
        "packages":{name:importlib.metadata.version(name) for name in package_names},
        "processor_class":type(processor).__name__, "image_processor":processor.image_processor.to_dict(),
        "chat_template":processor.chat_template, "warmup_runs":1,
        "started_at_utc":datetime.now(timezone.utc).isoformat(), "complete":False, "results":[]}
    save(output,meta)
    for i,q in enumerate(questions,1):
        with Image.open(image_path(q,args.benchmark)) as original:
            image = original.convert("RGB")
            tick = time.perf_counter()
            result = generate_answer(model,processor,image,q["prompt"],generation)
            result["preprocess_and_generation_seconds"] = time.perf_counter()-tick
            result["image_dimensions"] = list(image.size)
        meta["results"].append({**request_identity(q),**result})
        meta["complete"] = i == len(questions)
        save(output,meta)
        print(f"[{i:02d}/20 {q['id']}] {result['answer']} ({result['generation_seconds']:.2f}s)",flush=True)
    print("Saved: "+str(output.resolve()),flush=True)


if __name__ == "__main__":
    main()
