# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

import mmcv
import torch
from torch import Tensor

from mmcv.runner import load_checkpoint

from mmedit.models import build_model

import PIL
import cv2
from PIL import Image


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
        # torch.save({'model': model.state_dict()}, 'keys.pth')

        keys_1 = params['model'].keys()
        print(keys_0 == keys_1)

        model.load_state_dict(params['model'])

    # model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def pil2tensor(image, dtype):
    "Convert PIL style `image` array to torch style image tensor."
    a = np.asarray(image)
    if a.ndim == 2: a = np.expand_dims(a, 2)
    a = np.transpose(a, (1, 0, 2))
    a = np.transpose(a, (2, 1, 0))
    return torch.from_numpy(a.astype(dtype, copy=False) )


def norm(x, mean, std):
    x = (x - mean[..., None, None]) / std[..., None, None]
    return x


def denorm(x, mean, std):
    x = x * std[..., None, None] + mean[..., None, None]
    return x


def post_process(raw_color, orig):
    color_np = np.asarray(raw_color)
    orig_np = np.asarray(orig)
    color_yuv = cv2.cvtColor(color_np, cv2.COLOR_BGR2YUV)
    # do a black and white transform first to get better luminance values
    orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_BGR2YUV)
    hires = np.copy(orig_yuv)
    hires[:, :, 1:3] = color_yuv[:, :, 1:3]
    final = cv2.cvtColor(hires, cv2.COLOR_YUV2BGR)
    final = Image.fromarray(final)
    return final


def colorization_inference(model, img_path):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img (str): File path of input image.

    Returns:
        np.ndarray: The predicted colorization result.
    """
    # # cfg = model.cfg
    # # device = next(model.parameters()).device  # model device
    #
    # # prepare data
    # data = torch.load(img, map_location='cpu')
    # x, y, res, out = data.values()
    # x_ = x.unsqueeze(0).cuda()
    #
    # # forward the model
    # with torch.no_grad():
    #     results = model.forward(x_).squeeze()
    torch.cuda.empty_cache()
    orig_image = PIL.Image.open(img_path).convert('RGB')
    render_factor = 10
    render_base = 16
    render_sz = render_factor * render_base
    targ_sz = (render_sz, render_sz)
    model_image = orig_image.resize(targ_sz, resample=PIL.Image.BILINEAR).convert('LA').convert('RGB')
    x = pil2tensor(model_image, np.float32)
    x = x.cuda()

    x.div_(255)

    # imagenet的均值和方差
    mean = torch.tensor([0.4850, 0.4560, 0.4060]).cuda()
    std = torch.tensor([0.2290, 0.2240, 0.2250]).cuda()

    x_ = norm(x, mean, std)


    model.eval()
    with torch.no_grad():
        results = model.forward(x_.unsqueeze(0)).squeeze().cpu()

    # results = torch.load('/home/SENSETIME/renqin/PycharmProjects/mmediting-master/res.pt').squeeze(0)
    # results 比 results_1小，理论上应该相等

    results = denorm(results.detach().cpu(), mean.cpu(), std.cpu())

    results = results.float().clamp(min=0, max=1)

    out = (results.numpy()*255).astype('uint8').transpose(1, 2, 0)
    out = Image.fromarray(out)

    # return PIL.fromarray(out)

    raw_color = out.resize(orig_image.size, resample=PIL.Image.BILINEAR)
    # import numpy as np
    print(np.array(raw_color))
    final = post_process(raw_color, orig_image)
    print(np.array(final))

    return final
