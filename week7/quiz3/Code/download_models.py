"""Optional: recreate pinned base-model caches on a different machine."""
from common import MODELS


def main():
    import truststore
    truststore.inject_into_ssl()
    from huggingface_hub import snapshot_download
    for kind,item in MODELS.items():
        snapshot_download(repo_id=item['model_id'],revision=item['revision'],cache_dir=item['cache'],
            allow_patterns=['*.safetensors','*.json','*.txt','*.model','*.jinja','*.tiktoken'])
        print('Cached pinned base model: '+kind,flush=True)


if __name__=='__main__':main()
