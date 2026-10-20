from __future__ import annotations

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms


class ImageBackbone(nn.Module):
    def __init__(self, kind, pretrained=True):
        super().__init__()
        self.kind = kind
        if kind == "resnet50":
            self.net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            self.net.fc = nn.Identity()
            self.dimension = 2048
        elif kind == "vit_b_16":
            self.net = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
            self.net.heads = nn.Identity()
            self.dimension = 768
        else:
            raise ValueError(kind)

    def forward(self, images):
        return self.net(images)


class BirdImages(Dataset):
    def __init__(self, root, records, train=False, seed=42):
        self.root, self.records, self.train, self.seed = root, records, train, seed
        self.transform = transforms.Compose([
            *([transforms.RandomResizedCrop(224, scale=(.65, 1.0)), transforms.RandomHorizontalFlip()]
              if train else [transforms.Resize(256), transforms.CenterCrop(224)]),
            transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        epoch = 0
        if isinstance(index, tuple):
            index, epoch = index[:2]
        row = self.records[index]
        with Image.open(self.root / row["image"]) as image:
            # Epoch is sent in each sampler index; persistent Windows workers see it too.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed + epoch * 1000003 + index)
                image = self.transform(image.convert("RGB"))
        return image, row["label"]
