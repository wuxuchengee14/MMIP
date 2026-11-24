"""Shared benchmark identities and atomic JSON I/O; no GPU dependencies."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "smol": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
}
GENERATION = {"max_new_tokens": 256, "do_sample": False, "num_beams": 1, "use_cache": True}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def image_path(question, benchmark_path):
    base = Path(benchmark_path).resolve().parent
    path = (base / question["image"]).resolve()
    require(path.is_relative_to(base), "Image path must remain inside the benchmark directory")
    return path


def load_benchmark(path):
    questions = read(path)
    require(isinstance(questions, list) and len(questions) == 20, "Benchmark must contain exactly 20 questions")
    require(len({q["id"] for q in questions}) == 20, "Question IDs must be unique")
    require(Counter(q["grading"]["method"] for q in questions)=={"exact":14,"ocr":2,"bbox":2,"caption":2}, "Expected 14 exact, 2 OCR, 2 grounding and 2 caption questions")
    require(len({q["category"] for q in questions})==10 and all(n==2 for n in Counter(q["category"] for q in questions).values()),"Expected 10 capabilities with 2 questions each")
    for q in questions:
        require(isinstance(q["prompt"], str) and q["prompt"].strip(), "Empty prompt")
        require(sha(image_path(q, path)) == q["image_sha256"], "Image changed: " + q["id"])
        require(q["grading"]["method"] in {"exact", "ocr", "bbox", "caption"}, "Unknown grading method")
    manifest = read(Path(path).with_name("benchmark_manifest.json"))
    require(manifest["benchmark_sha256"] == sha(path), "Benchmark file changed; regenerate it deliberately")
    require(manifest["task_digest"] == digest(questions), "Benchmark manifest mismatch")
    require(manifest["source_manifest_sha256"] == sha(Path(path).with_name("sources.json")), "Source manifest changed")
    return questions


def request_identity(question):
    return {"id": question["id"], "image_sha256": question["image_sha256"], "prompt": question["prompt"]}
