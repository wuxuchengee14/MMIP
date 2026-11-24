"""Local Qwen inference with reproducible settings and optional GPU profiling."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3-8B"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=["bf16", "int8", "nf4"], default="bf16")
    parser.add_argument("--revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    parser.add_argument("--prompt", default="請用繁體中文，以三個要點說明什麼是大型語言模型。")
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--sample-ms", type=float, default=50)
    parser.add_argument("--nvml-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    memory_path = args.output.with_name(args.output.stem + ".memory.json")
    if not args.overwrite and (args.output.exists() or memory_path.exists()):
        parser.error("Output already exists. Choose a new --output filename, or intentionally use --overwrite.")
    if args.sample_ms <= 0 or args.max_new_tokens <= 0:
        parser.error("sample-ms and max-new-tokens must be positive")
    questions = (json.loads(args.questions.read_text(encoding="utf-8")) if args.questions
                 else [{"id": "demo", "prompt": args.prompt}])
    if not questions or len({q["id"] for q in questions}) != len(questions):
        parser.error("Questions must have unique IDs and must not be empty")
    prompt_key = "optimized_prompt" if args.optimized else "prompt"
    if any(not isinstance(q.get(prompt_key), str) or not q[prompt_key].strip() for q in questions):
        parser.error(f"Every question needs a nonempty {prompt_key}")

    import torch
    import truststore
    truststore.inject_into_ssl()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; check the active Conda environment.")
    monitor = None
    try:
        if args.profile:
            from gpu_memory import NvidiaMemory
            monitor = NvidiaMemory(args.nvml_index)
            idle_bytes = monitor.read()["device_used_bytes"]

        load_options = {
            "revision": args.revision, "cache_dir": str(ROOT / "models"),
            "torch_dtype": torch.bfloat16, "device_map": {"": 0},
            "low_cpu_mem_usage": True,
        }
        if args.precision == "int8":
            load_options["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif args.precision == "nf4":
            load_options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

        print("Loading Qwen. The first run downloads model weights.", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=args.revision, cache_dir=str(ROOT / "models"))
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_options).eval()
        actual_revision = getattr(model.config, "_commit_hash", None)
        if actual_revision is None:
            raise RuntimeError("Cannot record the loaded model revision")
        quant4, quant8 = bool(getattr(model, "is_loaded_in_4bit", False)), bool(getattr(model, "is_loaded_in_8bit", False))
        if quant4 != (args.precision == "nf4") or quant8 != (args.precision == "int8"):
            raise RuntimeError("Loaded quantization flags disagree with requested precision")
        generation = {
            "max_new_tokens": args.max_new_tokens, "do_sample": True,
            "temperature": 0.7, "top_p": 0.8, "top_k": 20,
            "pad_token_id": tokenizer.eos_token_id,
        }
        system = "請完成使用者任務，遵守題目指定的語言與輸出格式。"

        def encode(prompt):
            return tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", enable_thinking=False).to("cuda:0")

        # One fixed warmup, excluded from timing and memory windows for all groups.
        set_seed(41)
        with torch.inference_mode():
            model.generate(**encode("請只回答「測試」。"), **dict(generation, max_new_tokens=8))
        torch.cuda.synchronize()

        meta = {
            "schema_version": 2, "model": MODEL_ID, "revision": actual_revision,
            "precision": args.precision, "optimized": args.optimized,
            "thinking": False, "loaded_in_4bit": quant4, "loaded_in_8bit": quant8,
            "generation_config": generation, "system_prompt": system,
            "seed_policy": "42 + question index",
            "task_sha256": digest([{"id":q["id"], "prompt":q.get("basic_prompt",q.get("prompt")),
                                  "reference":q.get("reference"), "grading":q.get("grading")} for q in questions]),
            "selected_prompts_sha256": digest([q[prompt_key] for q in questions]),
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "warmup_generations": 1,
            "profile_enabled": args.profile,
        }
        results, windows = [], []
        profile = None
        if monitor:
            profile = {
                "schema_version": 1, "model": MODEL_ID, "revision": actual_revision,
                "precision": args.precision, "optimized": args.optimized,
                "gpu": monitor.name, "gpu_uuid": monitor.uuid,
                "nvml_index": args.nvml_index, "sample_interval_ms": args.sample_ms,
                "idle_device_used_bytes_before_loading": idle_bytes,
                "note": "NVML is whole-device usage including desktop/other apps; PyTorch metrics are allocator-specific. Idle subtraction is an estimate.",
                "windows": windows,
            }
        print(f"GPU: {meta['gpu']}; precision={args.precision}; optimized={args.optimized}", flush=True)
        for index, question in enumerate(questions):
            seed = 42 + index
            set_seed(seed)
            prompt = question[prompt_key]
            inputs = encode(prompt)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(0)
            window = None
            if monitor:
                from gpu_memory import MemoryWindow
                window = MemoryWindow(monitor.read, args.sample_ms/1000).start()
            tick = time.perf_counter()
            try:
                with torch.inference_mode():
                    generated = model.generate(**inputs, **generation)
                torch.cuda.synchronize()
                seconds = time.perf_counter()-tick
            finally:
                measured = window.stop() if window else None

            new_tokens = generated[0, inputs["input_ids"].shape[1]:]
            token_ids = new_tokens.cpu().tolist()
            eos = model.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            terminated = bool(token_ids and token_ids[-1] in eos_ids)
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            row = {
                **question, "used_prompt": prompt, "answer": answer, "seed": seed,
                "input_tokens": inputs["input_ids"].shape[1],
                "output_tokens": len(new_tokens), "generation_seconds": seconds,
                "tokens_per_second_including_prefill": len(new_tokens)/seconds,
                "hit_token_limit": len(new_tokens) >= args.max_new_tokens and not terminated,
                "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            }
            if measured:
                measured.update(id=question["id"], torch_peak_allocated_bytes=row["torch_peak_allocated_bytes"])
                windows.append(measured)
                row["memory"] = measured["summary"]
                from gpu_memory import aggregate
                profile["summary"] = aggregate(windows)
                profile["summary"]["estimated_device_bytes_above_idle_mean"] = profile["summary"]["device_used_bytes_mean"]-idle_bytes
                write_json(memory_path, profile)
            results.append(row)
            write_json(args.output, {**meta, "results": results,
                "memory_file": str(memory_path.resolve()) if monitor else None,
                "memory_summary": profile["summary"] if profile else None})
            print(f"[{question['id']}] {answer}", flush=True)
        print(f"Saved: {args.output.resolve()}", flush=True)
    finally:
        if monitor:
            monitor.close()


if __name__ == "__main__":
    main()
