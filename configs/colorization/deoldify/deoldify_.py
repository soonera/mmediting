from mmedit.models import resnet101


model = dict(
    type='DeOldify',
    encoder=resnet101(pretrained=False),
    n_classes=3,
    blur=True,
    blur_final=True,
    self_attention=True,
    y_range=(-3.0, 3.0),
    last_cross=True,
    bottle=False,
    nf_factor=2
    )

train_cfg = dict()
test_cfg = dict()

