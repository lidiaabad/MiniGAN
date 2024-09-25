# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import math
import torch
import torch as th
from torch.functional import Tensor
from torch.nn import functional as F


def sinc(t):
    """sinc.
    :param t: the input tensor
    """
    return th.where(t == 0, th.tensor(1., device=t.device, dtype=t.dtype), th.sin(t) / t)


def kernel_upsample2(zeros: int=56) -> torch.Tensor:
    """kernel_upsample2.
    """
    win = th.hann_window(4 * zeros + 1, dtype=torch.float, periodic=False)
    winodd = win[1::2]
    t = th.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t *= math.pi
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


@torch.jit.script
def upsample2(x: torch.Tensor, zeros: int=56):
    """
    Upsampling the input by 2 using sinc interpolation.
    Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
    ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
    Vol. 9. IEEE, 1984.
    """
    #*other, time = x.shape
    b, c, time = x.shape
    kernel = kernel_upsample2(zeros).to(x)
    #out = F.conv1d(x.view(-1, 1, time), kernel, padding=zeros)[..., 1:].view(*other, time)
    out = F.conv1d(x.view(-1, 1, time), kernel, padding=zeros)[..., 1:].view(b, c, time)
    y = th.stack([x, out], dim=-1)
    #return y.view(*other, -1)
    return y.view(b, c, -1)


def kernel_downsample2(zeros: int=56) -> torch.Tensor:
    """kernel_downsample2.
    """
    win = th.hann_window(4 * zeros + 1, dtype=torch.float, periodic=False)
    winodd = win[1::2]
    t = th.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t.mul_(math.pi)
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


#@torch.jit.export # this also works
@torch.jit.script
def downsample2(x: torch.Tensor, zeros: int=56):
    """
    Downsampling the input by 2 using sinc interpolation.
    Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
    ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
    Vol. 9. IEEE, 1984.
    """
    if x.shape[-1] % 2 != 0:
        x = F.pad(x, (0, 1))
    xeven = x[..., ::2]
    xodd = x[..., 1::2]
    #*other, time = xodd.shape
    b, c, time = xodd.shape
    kernel = kernel_downsample2(zeros).to(x)
    out = xeven + F.conv1d(xodd.view(-1, 1, time), kernel, padding=zeros)[..., :-1].view(
    #    *other, time)
         b, c, time)
    #return out.view(*other, -1).mul(0.5)
    return out.view(b, c, -1).mul(0.5)
