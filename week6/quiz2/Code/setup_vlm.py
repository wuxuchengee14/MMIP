"""Check the active environment and freeze Hugging Face model revisions (no weights)."""
import argparse
import re

from common import MODELS, ROOT, read, require, save


def check_environment():
    import torch
    import torchvision
    import PIL
    import num2words
    import accelerate
    import transformers
    from transformers import Qwen2_5_VLForConditionalGeneration, SmolVLMForConditionalGeneration, SmolVLMProcessor
    require(transformers.__version__ == "4.57.6", "Use transformers==4.57.6 for these scripts")
    require(torch.__version__.startswith("2.9.1"), "Expected torch 2.9.1; keep it matched with torchvision 0.24.1")
    require(torchvision.__version__.startswith("0.24.1"), "Expected torchvision 0.24.1")
    require(torch.cuda.is_available(), "CUDA unavailable")
    require(torch.cuda.is_bf16_supported(), "GPU must support BF16")
    print("Python / torch / torchvision / transformers / GPU:", flush=True)
    import sys
    print(sys.executable, torch.__version__, torchvision.__version__, transformers.__version__, torch.cuda.get_device_name(0), sep="\n", flush=True)
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.2f} GiB", flush=True)
    # Exercise a small CUDA operation, including the current GPU architecture.
    with torch.inference_mode():
        x = torch.ones((64,64), device="cuda", dtype=torch.bfloat16)
        require(float((x@x)[0,0]) == 64, "CUDA check failed")
    print("PASS: dependencies, vision processor imports and BF16 CUDA matmul.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--refresh-lock", action="store_true")
    parser.add_argument("--check-processors", action="store_true", help="Also download small processor/tokenizer files and encode all 20 images on CPU")
    args = parser.parse_args()
    check_environment()
    if args.check_only:
        return
    import truststore
    truststore.inject_into_ssl()
    from huggingface_hub import HfApi
    lock_path = ROOT/"Data/models.lock.json"
    if lock_path.exists() and not args.refresh_lock:
        lock = read(lock_path)
        print("Keeping existing pinned model revisions.", flush=True)
    else:
        records = {}
        for key, model_id in MODELS.items():
            info = HfApi().model_info(model_id)
            require(bool(re.fullmatch(r"[a-f0-9]{40}", info.sha or "")), "Cannot resolve model commit")
            records[key] = {"model_id":model_id, "revision":info.sha}
            print(f"{key}: {model_id}@{info.sha}", flush=True)
        lock = {"schema_version":1, "models":records}
        save(lock_path,lock)
    for key, model_id in MODELS.items():
        require(lock["models"][key]["model_id"] == model_id, "Unexpected model in lock")
    if args.check_processors:
        from common import image_path, load_benchmark
        from run_vlm import encode, load_processor
        from PIL import Image
        benchmark = ROOT/"Data/benchmark.json"
        questions = load_benchmark(benchmark)
        results = {}
        for key, item in lock["models"].items():
            processor = load_processor(key, item["revision"])
            counts = []
            for q in questions:
                with Image.open(image_path(q,benchmark)) as im:
                    inputs, rendered = encode(processor, im.convert("RGB"), q["prompt"])
                require("pixel_values" in inputs and "input_ids" in inputs, "No encoded image tokens")
                counts.append({"id":q["id"], "input_tokens":inputs["input_ids"].shape[-1],
                               "pixel_shape":list(inputs["pixel_values"].shape)})
            results[key] = {**item, "encoded":counts}
            print(f"PASS: {key} processor encoded all 20 images without model weights.", flush=True)
        save(ROOT/"Output/processor_check.json",results)


if __name__ == "__main__":
    main()
