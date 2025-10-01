from torch.nn import Module
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import Dropout
from torch.nn import LayerNorm
from torch.nn import Identity


def drop_path(x, drop_prob: 'float'=0.0, training: 'bool'=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.
        device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class TransformerEncoderLayer_MLP(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
        attention_dropout=0.0, drop_path_rate=0.5, layerscale=0.0,
        train_scale=True):
        super(TransformerEncoderLayer_MLP, self).__init__()
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)
        self.drop_path = DropPath(drop_path_rate
            ) if drop_path_rate > 0 else Identity()
        self.activation = F.gelu
        self.layerscale = layerscale
        if layerscale > 0.0:
            if train_scale:
                self.gamma = nn.Parameter(layerscale * torch.ones(d_model),
                    requires_grad=True)
            else:
                self.gamma = nn.Parameter(layerscale * torch.ones(d_model),
                    requires_grad=False)

    def forward(self, src: 'torch.Tensor', *args, **kwargs) ->torch.Tensor:
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        if self.layerscale > 0.0:
            src = src + self.drop_path(self.gamma * self.dropout2(src2))
        else:
            src = src + self.drop_path(self.dropout2(src2))
        return src


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'nhead': 4}]
