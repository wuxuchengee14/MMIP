import argparse
import importlib
import json
import platform
import sys
from pathlib import Path

import torch
import torchvision

from common.utils import ROOT, write_json


def check(device="cuda", output=True):
    for package in ("numpy", "sklearn", "matplotlib", "PIL", "medmnist", "nbformat", "truststore"):
        importlib.import_module(package)
    information = {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda, "device": device,
    }
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Install environment.yml and activate MMIP.")
        information.update(gpu=torch.cuda.get_device_name(0),
                           capability=list(torch.cuda.get_device_capability(0)),
                           vram_gib=torch.cuda.get_device_properties(0).total_memory / 1024**3,
                           architectures=torch.cuda.get_arch_list())
        model = torch.nn.Conv2d(3, 8, 3).cuda()
        opt = torch.optim.AdamW(model.parameters())
        scaler = torch.amp.GradScaler("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(torch.randn(2, 3, 32, 32, device="cuda")).float().square().mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        information["amp_backward"] = "passed"
    print(json.dumps(information, indent=2, ensure_ascii=False), flush=True)
    if output:
        write_json(ROOT / "hardware.json", information)
    return information


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    check(parser.parse_args().device)
