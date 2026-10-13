"""Check Windows spawn workers with the production raw-image Dataset."""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from common.utils import loader
from quiz3.Code.train import OrganDataset, preprocess


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        file = Path(tmp)/"data.npz"
        images = np.arange(4*28*28).reshape(4,28,28).astype("uint8")
        np.savez(file, train_images=images, train_labels=np.arange(4)[:,None])
        dataset = OrganDataset(file, "train")
        batches = loader(dataset, {"workers": 2, "pin_memory": True}, 2, shuffle=False)
        count = 0
        for raw, labels in batches:
            result = preprocess(raw, torch.device("cuda"))
            assert result.shape == (2,3,224,224)
            assert torch.isfinite(result).all()
            count += len(labels)
        assert count == 4
        del batches
        print("Windows spawn workers + pinned raw uint8 + CUDA preprocessing: PASS")
