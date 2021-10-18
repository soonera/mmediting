# model settings
from fastai.vision import models
from fastai.vision.learner import create_body
# from fastai.layers import NormType


model = dict(
    type='DeOldify',
    encoder=create_body(arch=models.resnet101, pretrained=True),
    # encoder='ResNet101',
    n_classes=3,
    blur=True,
    blur_final=True,
    self_attention=True,
    y_range=(-3.0, 3.0),
    last_cross=True,
    bottle=False,
    # norm_type="NormSpectral",
    # norm_type=NormType.Spectral,
    nf_factor=2
    # backbone=dict(
    #     type='SimpleEncoderDecoder',
    #     encoder=dict(type='VGG16', in_channels=4),
    #     decoder=dict(type='PlainDecoder')),
    # pretrained='open-mmlab://mmedit/vgg16',
    # loss_alpha=dict(type='CharbonnierLoss', loss_weight=0.5),
    # loss_comp=dict(type='CharbonnierCompLoss', loss_weight=0.5)
    )

test_cfg = dict()

