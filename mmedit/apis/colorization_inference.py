# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

import mmcv
import torch

from mmcv.runner import load_checkpoint

from mmedit.models import build_model


def init_colorization_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Which device the model will deploy. Default: 'cuda:0'.

    Returns:
        nn.Module: The constructed model.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    # config.model.pretrained = None
    # config.test_cfg.metrics = None
    model = build_model(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        # checkpoint = load_checkpoint(model, checkpoint)
        params = torch.load(checkpoint, map_location='cpu')

        # 自己加的
        keys_0 = model.state_dict().keys()
        keys_1 = params['model'].keys()
        print(keys_0 == keys_1)

        model.load_state_dict(params['model'])

    # model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def colorization_inference(model, img):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img (str): File path of input image.

    Returns:
        np.ndarray: The predicted colorization result.
    """
    # cfg = model.cfg
    # device = next(model.parameters()).device  # model device

    # prepare data
    data = torch.load(img, map_location='cpu')
    x, y, res, out = data.values()
    x_ = x.unsqueeze(0).cuda()

    # forward the model
    with torch.no_grad():
        results = model.forward(x_).squeeze()

    return results
