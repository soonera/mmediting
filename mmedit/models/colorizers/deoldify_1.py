# 全改了，但是后面几个keys不对

from fastai.callbacks.hooks import Hook, model_sizes, hook_outputs, dummy_eval

from typing import List, Tuple

from fastai.vision import in_channels, Sizes
# from fastai.layers import NormType
from fastai.torch_core import Optional, Callable, tensor

import numpy as np
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.weight_norm import weight_norm
from torch.nn.utils.spectral_norm import spectral_norm

import torch
from ..registry import MODELS


# √
def relu(inplace: bool = False, leaky: float = None):
    "Return a relu activation, maybe `leaky` and `inplace`."
    return nn.LeakyReLU(inplace=inplace, negative_slope=leaky) if leaky is not None else nn.ReLU(inplace=inplace)


# √
class SequentialEx(nn.Module):
    "Like `nn.Sequential`, but with ModuleList semantics, and can access module input"

    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        res = x
        for l in self.layers:
            res.orig = x
            nres = l(res)
            # We have to remove res.orig to avoid hanging refs and therefore memory leaks
            res.orig = None
            res = nres
        return res

    def __getitem__(self, i): return self.layers[i]

    def append(self, l): return self.layers.append(l)

    def extend(self, l): return self.layers.extend(l)

    def insert(self, i, l): return self.layers.insert(i, l)


# √
def conv1d(ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return spectral_norm(conv)


# √
def conv_layer(ni: int, nf: int, ks: int = 3, stride: int = 1, padding: int = None, bias: bool = None,
               is_1d: bool = False,
               # norm_type: Optional[NormType] = NormType.Batch, use_activ: bool = True, leaky: float = None,
               norm_type: str = "NormBatch", use_activ: bool = True, leaky: float = None,
               transpose: bool = False, init: Callable = nn.init.kaiming_normal_, self_attention: bool = False):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."
    if padding is None:
        padding = (ks - 1) // 2 if not transpose else 0
    bn = norm_type in ("NormBatch", "NormBatchZero")
    if bias is None:
        bias = not bn
    conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv2d
    conv = init_default(conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding), init)
    if norm_type == "NormWeight":
        conv = weight_norm(conv)
    elif norm_type == "NormSpectral":
        conv = spectral_norm(conv)
    layers = [conv]
    if use_activ:
        layers.append(relu(True, leaky=leaky))
    if bn:
        layers.append((nn.BatchNorm1d if is_1d else nn.BatchNorm2d)(nf))
    if self_attention:
        layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)


# √
class MergeLayer(nn.Module):
    "Merge a shortcut with the result of the module by adding them or concatenating thme if `dense=True`."

    def __init__(self, dense: bool = False):
        super().__init__()
        self.dense = dense

    def forward(self, x): return torch.cat([x, x.orig], dim=1) if self.dense else (x + x.orig)


# √
def res_block(nf, dense: bool = False, norm_type: str = "NormBatch", bottle: bool = False,
              **conv_kwargs):
    "Resnet block of `nf` features. `conv_kwargs` are passed to `conv_layer`."
    norm2 = norm_type
    if not dense and (norm_type == "NormBatch"):
        norm2 = "BatchZero"
    nf_inner = nf // 2 if bottle else nf
    return SequentialEx(conv_layer(nf, nf_inner, norm_type=norm_type, **conv_kwargs),
                        conv_layer(nf_inner, nf, norm_type=norm2, **conv_kwargs),
                        MergeLayer(dense))


def init_default(m: nn.Module, func=nn.init.kaiming_normal_) -> nn.Module:
    "Initialize `m` weights with `func` and set `bias` to 0."
    if func:
        if hasattr(m, 'weight'):
            func(m.weight)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.)
    return m


# √
class SelfAttention(nn.Module):
    "Self attention layer for nd."

    def __init__(self, n_channels: int):
        super().__init__()
        self.query = conv1d(n_channels, n_channels // 8)
        self.key = conv1d(n_channels, n_channels // 8)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(tensor([0.]))

    def forward(self, x):
        # Notation from https://arxiv.org/pdf/1805.08318.pdf
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


def sigmoid_range(x, low, high):
    "Sigmoid function with range `(low, high)`"
    return torch.sigmoid(x) * (high - low) + low


class SigmoidRange(nn.Module):
    "Sigmoid module with range `(low,x_max)`"

    def __init__(self, low, high):
        super().__init__()
        self.low, self.high = low, high

    def forward(self, x): return sigmoid_range(x, self.low, self.high)


def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function."
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale ** 2))
    k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale ** 2)
    k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
    x.data.copy_(k)


class PixelShuffle_ICNR(nn.Module):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."

    def __init__(self, ni: int, nf: int = None, scale: int = 2, blur: bool = False, norm_type: str = "NormWeight",
                 leaky: float = None):
        super().__init__()
        if not nf:
            nf = ni
        # nf = ifnone(nf, ni)
        self.conv = conv_layer(ni, nf * (scale ** 2), ks=1, norm_type=norm_type, use_activ=False)
        icnr(self.conv[0].weight)
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = relu(True, leaky=leaky)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.blur else x


def custom_conv_layer(
        ni: int,
        nf: int,
        ks: int = 3,
        stride: int = 1,
        padding: int = None,
        bias: bool = None,
        is_1d: bool = False,
        # norm_type: Optional[NormType] = NormType.Batch,
        norm_type: str = "NormBatch",
        use_activ: bool = True,
        leaky: float = None,
        transpose: bool = False,
        init: Callable = nn.init.kaiming_normal_,
        self_attention: bool = False,
        extra_bn: bool = False,
):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."
    if padding is None:
        padding = (ks - 1) // 2 if not transpose else 0
    # bn = norm_type in (NormType.Batch, NormType.BatchZero) or extra_bn == True
    bn = norm_type in ("NormBatch", "NormBatchZero") or extra_bn == True

    if bias is None:
        bias = not bn

    conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv2d

    conv = init_default(
        conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding),
        init,
    )

    if norm_type == "NormWeight":
        conv = weight_norm(conv)
    elif norm_type == "NormSpectral":
        conv = spectral_norm(conv)

    layers = [conv]
    if use_activ:
        layers.append(relu(True, leaky=leaky))
    if bn:
        layers.append((nn.BatchNorm1d if is_1d else nn.BatchNorm2d)(nf))
    if self_attention:
        layers.append(SelfAttention(nf))

    return nn.Sequential(*layers)


def _get_sfs_idxs(sizes: Sizes) -> List[int]:
    "Get the indexes of the layers where the size of the activation changes."
    feature_szs = [size[-1] for size in sizes]
    sfs_idxs = list(
        np.where(np.array(feature_szs[:-1]) != np.array(feature_szs[1:]))[0]
    )
    if feature_szs[0] != feature_szs[1]:
        sfs_idxs = [0] + sfs_idxs
    return sfs_idxs


class CustomPixelShuffle_ICNR(nn.Module):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."

    def __init__(
            self,
            ni: int,
            nf: int = None,
            scale: int = 2,
            blur: bool = False,
            leaky: float = None,
            **kwargs
    ):
        super().__init__()
        if not nf:
            nf = ni
        # nf = ifnone(nf, ni)
        self.conv = custom_conv_layer(
            ni, nf * (scale ** 2), ks=1, use_activ=False, **kwargs
        )
        icnr(self.conv[0].weight)
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)

        self.relu = relu(True, leaky=leaky)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.blur else x


class UnetBlockWide(nn.Module):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."

    def __init__(
            self,
            up_in_c: int,
            x_in_c: int,
            n_out: int,
            hook: Hook,
            final_div: bool = True,
            blur: bool = False,
            leaky: float = None,
            self_attention: bool = False,
            **kwargs
    ):
        super().__init__()
        self.hook = hook
        up_out = x_out = n_out // 2
        self.shuf = CustomPixelShuffle_ICNR(
            up_in_c, up_out, blur=blur, leaky=leaky, **kwargs
        )
        # self.bn = batchnorm_2d(x_in_c)
        self.bn = nn.BatchNorm2d(x_in_c)
        ni = up_out + x_in_c
        self.conv = custom_conv_layer(
            ni, x_out, leaky=leaky, self_attention=self_attention, **kwargs
        )
        self.relu = relu(leaky=leaky)

    def forward(self, up_in: Tensor) -> Tensor:
        s = self.hook.stored
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv(cat_x)


@MODELS.register_module()
# class DynamicUnetWide(SequentialEx):
class DeOldify(SequentialEx):
    "Create a U-Net from a given architecture."

    def __init__(
            self,
            encoder: nn.Module,
            n_classes: int,
            blur: bool = False,
            blur_final=True,
            self_attention: bool = False,
            y_range: Optional[Tuple[float, float]] = None,  # SigmoidRange
            last_cross: bool = True,
            bottle: bool = False,
            # norm_type: Optional[NormType] = NormType.Batch,
            norm_type: str = "NormBatch",
            nf_factor: int = 1,
            **kwargs
    ):

        nf = 512 * nf_factor
        extra_bn = norm_type == "NormSpectral"
        imsize = (256, 256)
        sfs_szs = model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs], detach=False)
        x = dummy_eval(encoder, imsize).detach()

        ni = sfs_szs[-1][1]
        kwargs_0 = {}  # 自己加的
        middle_conv = nn.Sequential(
            custom_conv_layer(
                ni, ni * 2, norm_type=norm_type, extra_bn=extra_bn, **kwargs_0
            ),
            custom_conv_layer(
                ni * 2, ni, norm_type=norm_type, extra_bn=extra_bn, **kwargs_0
            ),
        ).eval()
        x = middle_conv(x)
        # layers = [encoder, batchnorm_2d(ni), nn.ReLU(), middle_conv]
        layers = [encoder, nn.BatchNorm2d(ni), nn.ReLU(), middle_conv]

        for i, idx in enumerate(sfs_idxs):
            not_final = i != len(sfs_idxs) - 1
            up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
            do_blur = blur and (not_final or blur_final)
            sa = self_attention and (i == len(sfs_idxs) - 3)

            n_out = nf if not_final else nf // 2

            unet_block = UnetBlockWide(
                up_in_c,
                x_in_c,
                n_out,
                self.sfs[i],
                final_div=not_final,
                blur=blur,
                self_attention=sa,
                norm_type=norm_type,
                extra_bn=extra_bn,
                **kwargs_0
            ).eval()
            layers.append(unet_block)
            x = unet_block(x)

        ni = x.shape[1]
        if imsize != sfs_szs[0][-2:]:
            layers.append(PixelShuffle_ICNR(ni, **kwargs_0))
        if last_cross:
            layers.append(MergeLayer(dense=True))
            ni += in_channels(encoder)
            layers.append(res_block(ni, bottle=bottle, **kwargs_0))
        layers += [
            custom_conv_layer(ni, n_classes, ks=1, use_activ=False)
        ]
        if y_range is not None:
            layers.append(SigmoidRange(*y_range))
        super().__init__(*layers)

    def __del__(self):
        if hasattr(self, "sfs"):
            self.sfs.remove()
