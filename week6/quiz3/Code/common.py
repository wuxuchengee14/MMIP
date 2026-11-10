"""Experiment definitions, immutable identities, and local model cache locations."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEEK = ROOT.parent
MODELS = {
    "llm": {"model_id":"Qwen/Qwen3-8B", "revision":"b968826d9c46dd6066d109eabc6255188de91218", "cache":str(WEEK/"quiz1/models"), "precision":"nf4"},
    "vlm": {"model_id":"HuggingFaceTB/SmolVLM2-2.2B-Instruct", "revision":"482adb537c021c86670beed01cd58990d01e72e4", "cache":str(WEEK/"quiz2/models"), "precision":"bf16"},
}
GENERATION = {"do_sample":False,"num_beams":1,"max_new_tokens":192,"repetition_penalty":1.0}
VARIANTS = {"original":{}, "jpeg_q15":{"jpeg_quality":15}, "lowres_64":{"longest_edge":64},
            "dark_050":{"brightness":0.5}, "bright_150":{"brightness":1.5}}


def require(condition,message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def data_path(relative):
    path = (ROOT/"Data"/relative).resolve()
    require(path.is_relative_to((ROOT/"Data").resolve()),"Path outside Data")
    return path


def validate_data():
    manifest = read(ROOT/"Data/manifest.json")
    for relative,expected in manifest["files"].items():
        require(sha(data_path(relative))==expected,"Dataset changed: "+relative)
    for kind in ["llm","vlm"]:
        train,dev,test = [read(ROOT/f"Data/{kind}_{split}.json") for split in ["train","dev","test"]]
        require(len(train)>=100 and len(dev)>0 and len(test)>0,"Insufficient data")
        prompts = [{digest({"prompt":q["prompt"],"image":q.get("image_sha256")}) for q in rows} for rows in [train,dev,test]]
        require(all(len(p)==len(rows) for p,rows in zip(prompts,[train,dev,test])),"Duplicate examples")
        require(not (prompts[0]&prompts[1] or prompts[0]&prompts[2] or prompts[1]&prompts[2]),"Train/dev/test overlap")
        for rows in [train,dev]:
            require(all(q["chosen"]!=q["rejected"] and q["chosen"].strip() and q["rejected"].strip() for q in rows),"Invalid preference pair")
        if kind=="vlm":
            images = [{q["image_sha256"] for q in rows} for rows in [train,dev,test]]
            require(not (images[0]&images[1] or images[0]&images[2] or images[1]&images[2]),"VLM image overlap")
    general = read(ROOT/"Data/general_100.json")
    require(len(general)==100 and len({q["prompt"] for q in general})==100,"Need 100 distinct unrelated questions")
    require(not ({q["prompt"] for q in general}&{q["prompt"] for q in read(ROOT/"Data/llm_train.json")}),"General benchmark overlaps training")
    return manifest
