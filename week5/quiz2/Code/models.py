from torch import nn
from common.vision import ImageBackbone


class FusionClassifier(nn.Module):
    def __init__(self, kind, classes, text_dim=768, fusion_dim=256, dropout=.3, pretrained=True):
        super().__init__()
        self.image_encoder = ImageBackbone(kind,pretrained)
        self.image_projection = nn.Sequential(nn.Linear(self.image_encoder.dimension,fusion_dim),nn.LayerNorm(fusion_dim),nn.GELU())
        self.text_projection = nn.Sequential(nn.Linear(text_dim,fusion_dim),nn.LayerNorm(fusion_dim),nn.GELU())
        self.classifier = nn.Sequential(nn.Linear(2*fusion_dim,fusion_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(fusion_dim,classes))

    def forward(self, images, text_features):
        import torch
        visual = self.image_projection(self.image_encoder(images))
        textual = self.text_projection(text_features)
        return self.classifier(torch.cat((visual,textual),dim=1))

    def optimizer_groups(self,settings):
        return [{"params":self.image_encoder.parameters(),"lr":settings["lr"]},
                {"params":[*self.image_projection.parameters(),*self.text_projection.parameters(),*self.classifier.parameters()],
                 "lr":settings.get("head_lr",settings["lr"])}]
