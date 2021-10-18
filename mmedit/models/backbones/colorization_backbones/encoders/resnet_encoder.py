import torch.nn as nn
from fastai.vision import models
from fastai.vision.learner import create_body
from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module()
class ResNet101(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet101 = create_body(arch=models.resnet101, pretrained=True)

    def forward(self, x):
        x = self.resnet101(x)
        return x