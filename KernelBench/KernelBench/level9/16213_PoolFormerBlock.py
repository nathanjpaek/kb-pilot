import torch
from torch import Tensor
from torch import nn


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Copied from timm
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """

    def __init__(self, p: 'float'=None):
        super().__init__()
        self.p = p

    def forward(self, x: 'Tensor') ->Tensor:
        if self.p == 0.0 or not self.training:
            return x
        kp = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = kp + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(kp) * random_tensor


class MLP(nn.Module):

    def __init__(self, dim, hidden_dim, out_dim=None) ->None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.act = nn.ReLU6(True)
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)

    def forward(self, x: 'Tensor') ->Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Pooling(nn.Module):

    def __init__(self, pool_size=3) ->None:
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, 1, pool_size // 2,
            count_include_pad=False)

    def forward(self, x: 'Tensor') ->Tensor:
        return self.pool(x) - x


class PoolFormerBlock(nn.Module):

    def __init__(self, dim, pool_size=3, dpr=0.0, layer_scale_init_value=1e-05
        ):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.token_mixer = Pooling(pool_size)
        self.norm2 = nn.GroupNorm(1, dim)
        self.drop_path = DropPath(dpr) if dpr > 0.0 else nn.Identity()
        self.mlp = MLP(dim, int(dim * 4))
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.
            ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.
            ones(dim), requires_grad=True)

    def forward(self, x: 'Tensor') ->Tensor:
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-
            1) * self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-
            1) * self.mlp(self.norm2(x)))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
