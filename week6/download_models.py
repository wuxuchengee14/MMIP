"""Download only the three pinned base-model checkpoints; weights remain gitignored."""
from pathlib import Path

ROOT=Path(__file__).resolve().parent
MODELS=[
    ('Qwen/Qwen3-8B','b968826d9c46dd6066d109eabc6255188de91218','quiz1/models'),
    ('Qwen/Qwen2.5-VL-7B-Instruct','cc594898137f460bfe9f0759e9844b3ce807cfb5','quiz2/models'),
    ('HuggingFaceTB/SmolVLM2-2.2B-Instruct','482adb537c021c86670beed01cd58990d01e72e4','quiz2/models'),
]


if __name__=='__main__':
    import truststore
    truststore.inject_into_ssl()
    from huggingface_hub import snapshot_download
    for model,revision,folder in MODELS:
        snapshot_download(repo_id=model,revision=revision,cache_dir=str(ROOT/folder),
            allow_patterns=['*.safetensors','*.json','*.txt','*.model','*.jinja','*.tiktoken'])
        print('Cached pinned model: '+model,flush=True)
