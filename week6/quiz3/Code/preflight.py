"""Validate compatible versions, cached weights and the CUDA training path."""
import importlib.metadata
from pathlib import Path
from common import MODELS,ROOT,require,validate_data


def main():
    import torch
    import bitsandbytes
    from trl import DPOTrainer
    from peft import LoraConfig
    for package,expected in {"transformers":"4.57.6","trl":"0.24.0","peft":"0.17.1","datasets":"4.3.0"}.items():
        actual=importlib.metadata.version(package);print(package,actual)
        require(actual==expected,f"Expected {package}=={expected}")
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),"CUDA BF16 unavailable")
    print(torch.cuda.get_device_name(0),f"{torch.cuda.get_device_properties(0).total_memory/2**30:.2f} GiB")
    validate_data()
    from models import processor_for
    for kind,item in MODELS.items():
        snapshot=Path(item['cache'])/('models--'+item['model_id'].replace('/','--'))/'snapshots'/item['revision']
        require(snapshot.is_dir() and any(snapshot.glob('*.safetensors')),'Cached weights missing: '+str(snapshot))
        processor_for(kind)
        print('PASS: cached model and processor',kind)
    print('PASS: frozen datasets, dependencies, cached weights and CUDA. No training or evaluations were run by this check.')


if __name__=='__main__':main()
