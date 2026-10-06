from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from common.data import load_arrays, make_transform
from common.utils import CLASSES, ROOT, load_config, seed_everything


def preview(config, root=ROOT):
    images, labels, _, _ = load_arrays(root)
    seed_everything(config["seed"])
    transform = make_transform("plain", config, augment=True)
    fig, axes = plt.subplots(11, 5, figsize=(10, 22))
    for k, name in enumerate(CLASSES):
        image = images[np.flatnonzero(labels == k)[0]]
        axes[k, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[k, 0].set_title(name)
        axes[k, 0].axis("off")
        for j in range(1, 5):
            augmented = transform(Image.fromarray(image))[0].numpy() * 0.5 + 0.5
            axes[k, j].imshow(augmented, cmap="gray", vmin=0, vmax=1)
            axes[k, j].set_title(f"Augmentation {j}")
            axes[k, j].axis("off")
    fig.tight_layout()
    output = Path(root) / "Quiz03_Generalization" / "Output"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "augmentation_examples.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preview train-only augmentation on development examples.")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    preview(load_config(args.config))
