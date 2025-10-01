import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from torch import optim as optim


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-06, data_format='channels_last'
        ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError
        self.normalized_shape = normalized_shape,

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self
                .bias, self.eps)
        elif self.data_format == 'channels_first':
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Minimum(nn.Module):

    def forward(self, x, y):
        return torch.minimum(x, y)


class NextMinBlock(nn.Module):
    """ ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        kernel_size (int): dws kernel_size
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-06,
        kernel_size=7):
        super().__init__()
        if kernel_size != 7:
            warnings.warn(f'Using kernel_size: {kernel_size}')
        self.dwconv_left = nn.Conv2d(dim, dim, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=dim)
        self.dwconv_right = nn.Conv2d(dim, dim, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=dim)
        self.instance_norm_relu = nn.Sequential(nn.InstanceNorm2d(dim), nn.
            ReLU())
        self.min = Minimum()
        self.norm = LayerNorm(dim, eps=1e-06)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path
            ) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x_left = self.dwconv_left(x)
        x_right = self.dwconv_right(x)
        x_left = self.instance_norm_relu(x_left)
        x_right = self.instance_norm_relu(x_right)
        x = self.min(x_left, x_right)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


class Abs(nn.Module):

    def forward(self, x):
        return torch.abs(x)


class NextMinMinusAbsBlockNoNorm(NextMinBlock):

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-06,
        kernel_size=7):
        super().__init__(dim, drop_path=drop_path, layer_scale_init_value=
            layer_scale_init_value, kernel_size=kernel_size)
        self.lambda_ = 2.0
        self.abs = Abs()
        self.instance_norm_relu = nn.Sequential(nn.ReLU())

    def forward(self, x):
        input = x
        x_left = self.dwconv_left(x)
        x_right = self.dwconv_right(x)
        x_left = self.instance_norm_relu(x_left)
        x_right = self.instance_norm_relu(x_right)
        x = self.lambda_ * self.min(x_left, x_right) - self.abs(x_left -
            x_right)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
