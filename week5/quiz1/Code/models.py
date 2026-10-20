import torch
from torch import nn
from torch.nn import functional as F
from common.vision import ImageBackbone


class RetrievalModel(nn.Module):
    def __init__(self, kind, classes, dimension=256, pretrained=True):
        super().__init__()
        self.encoder = ImageBackbone(kind, pretrained)
        self.projection = nn.Sequential(nn.Linear(self.encoder.dimension, 512), nn.ReLU(), nn.Linear(512, dimension))
        self.classifier = nn.Linear(dimension, classes)

    def forward(self, images):
        embeddings = F.normalize(self.projection(self.encoder(images)).float(), dim=1)
        return embeddings, self.classifier(embeddings)

    def optimizer_groups(self,settings):
        return [{"params":self.encoder.parameters(),"lr":settings["lr"]},
                {"params":[*self.projection.parameters(),*self.classifier.parameters()],
                 "lr":settings.get("head_lr",settings["lr"])}]


def supervised_contrastive(embeddings, labels, temperature=.1):
    logits = embeddings.float() @ embeddings.float().T / temperature
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives = labels[:, None].eq(labels[None]) & ~diagonal
    if not positives.any(1).all():
        raise ValueError("Each SupCon anchor must have a different same-class image in its batch")
    logits = logits.masked_fill(diagonal, -float("inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_prob.masked_fill(~positives, 0).sum(1) / positives.sum(1)).mean()
