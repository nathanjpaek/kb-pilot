import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.utils.data
from typing import Tuple
from typing import Optional
from typing import List
import torch.autograd


class EqualizedWeight(nn.Module):
    """
    <a id="equalized_weight"></a>
    ## Learning-rate Equalized Weights Parameter

    This is based on equalized learning rate introduced in the Progressive GAN paper.
    Instead of initializing weights at $\\mathcal{N}(0,c)$ they initialize weights
    to $\\mathcal{N}(0, 1)$ and then multiply them by $c$ when using it.
    $$w_i = c \\hat{w}_i$$

    The gradients on stored parameters $\\hat{w}$ get multiplied by $c$ but this doesn't have
    an affect since optimizers such as Adam normalize them by a running mean of the squared gradients.

    The optimizer updates on $\\hat{w}$ are proportionate to the learning rate $\\lambda$.
    But the effective weights $w$ get updated proportionately to $c \\lambda$.
    Without equalized learning rate, the effective weights will get updated proportionately to just $\\lambda$.

    So we are effectively scaling the learning rate by $c$ for these weight parameters.
    """

    def __init__(self, shape: 'List[int]'):
        """
        * `shape` is the shape of the weight parameter
        """
        super().__init__()
        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c


class EqualizedLinear(nn.Module):
    """
    <a id="equalized_linear"></a>
    ## Learning-rate Equalized Linear Layer

    This uses [learning-rate equalized weights]($equalized_weights) for a linear layer.
    """

    def __init__(self, in_features: 'int', out_features: 'int', bias:
        'float'=0.0):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        * `bias` is the bias initialization constant
        """
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: 'torch.Tensor'):
        return F.linear(x, self.weight(), bias=self.bias)


class Conv2dWeightModulate(nn.Module):
    """
    ### Convolution with Weight Modulation and Demodulation

    This layer scales the convolution weights by the style vector and demodulates by normalizing it.
    """

    def __init__(self, in_features: 'int', out_features: 'int', kernel_size:
        'int', demodulate: 'float'=True, eps: 'float'=1e-08):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        * `kernel_size` is the size of the convolution kernel
        * `demodulate` is flag whether to normalize weights by its standard deviation
        * `eps` is the $\\epsilon$ for normalizing
        """
        super().__init__()
        self.out_features = out_features
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2
        self.weight = EqualizedWeight([out_features, in_features,
            kernel_size, kernel_size])
        self.eps = eps

    def forward(self, x: 'torch.Tensor', s: 'torch.Tensor'):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `s` is style based scaling tensor of shape `[batch_size, in_features]`
        """
        b, _, h, w = x.shape
        s = s[:, None, :, None, None]
        weights = self.weight()[None, :, :, :, :]
        weights = weights * s
        if self.demodulate:
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4),
                keepdim=True) + self.eps)
            weights = weights * sigma_inv
        x = x.reshape(1, -1, h, w)
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)
        x = F.conv2d(x, weights, padding=self.padding, groups=b)
        return x.reshape(-1, self.out_features, h, w)


class StyleBlock(nn.Module):
    """
    <a id="style_block"></a>
    ### Style Block

    ![Style block](style_block.svg)

    *<small>$A$ denotes a linear layer.
    $B$ denotes a broadcast and scaling operation (noise is single channel).</small>*

    Style block has a weight modulation convolution layer.
    """

    def __init__(self, d_latent: 'int', in_features: 'int', out_features: 'int'
        ):
        """
        * `d_latent` is the dimensionality of $w$
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        """
        super().__init__()
        self.to_style = EqualizedLinear(d_latent, in_features, bias=1.0)
        self.conv = Conv2dWeightModulate(in_features, out_features,
            kernel_size=3)
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: 'torch.Tensor', w: 'torch.Tensor', noise:
        'Optional[torch.Tensor]'):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `w` is $w$ with shape `[batch_size, d_latent]`
        * `noise` is a tensor of shape `[batch_size, 1, height, width]`
        """
        s = self.to_style(w)
        x = self.conv(x, s)
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        return self.activation(x + self.bias[None, :, None, None])


class ToRGB(nn.Module):
    """
    <a id="to_rgb"></a>
    ### To RGB

    ![To RGB](to_rgb.svg)

    *<small>$A$ denotes a linear layer.</small>*

    Generates an RGB image from a feature map using $1 	imes 1$ convolution.
    """

    def __init__(self, d_latent: 'int', features: 'int'):
        """
        * `d_latent` is the dimensionality of $w$
        * `features` is the number of features in the feature map
        """
        super().__init__()
        self.to_style = EqualizedLinear(d_latent, features, bias=1.0)
        self.conv = Conv2dWeightModulate(features, 3, kernel_size=1,
            demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: 'torch.Tensor', w: 'torch.Tensor'):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `w` is $w$ with shape `[batch_size, d_latent]`
        """
        style = self.to_style(w)
        x = self.conv(x, style)
        return self.activation(x + self.bias[None, :, None, None])


class GeneratorBlock(nn.Module):
    """
    <a id="generator_block"></a>
    ### Generator Block

    ![Generator block](generator_block.svg)

    *<small>$A$ denotes a linear layer.
    $B$ denotes a broadcast and scaling operation (noise is a single channel).
    [*toRGB*](#to_rgb) also has a style modulation which is not shown in the diagram to keep it simple.</small>*

    The generator block consists of two [style blocks](#style_block) ($3 	imes 3$ convolutions with style modulation)
    and an RGB output.
    """

    def __init__(self, d_latent: 'int', in_features: 'int', out_features: 'int'
        ):
        """
        * `d_latent` is the dimensionality of $w$
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        """
        super().__init__()
        self.style_block1 = StyleBlock(d_latent, in_features, out_features)
        self.style_block2 = StyleBlock(d_latent, out_features, out_features)
        self.to_rgb = ToRGB(d_latent, out_features)

    def forward(self, x: 'torch.Tensor', w: 'torch.Tensor', noise:
        'Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]'):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `w` is $w$ with shape `[batch_size, d_latent]`
        * `noise` is a tuple of two noise tensors of shape `[batch_size, 1, height, width]`
        """
        x = self.style_block1(x, w, noise[0])
        x = self.style_block2(x, w, noise[1])
        rgb = self.to_rgb(x, w)
        return x, rgb


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_latent': 4, 'in_features': 4, 'out_features': 4}]
