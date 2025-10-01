import torch
import torch.nn as nn


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


class VanillaMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None,
        act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixerBlock(nn.Module):

    def __init__(self, num_patch, dim, token_mlp_ratio, channel_mlp_ratio,
        drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        token_mlp_dim = round(dim * token_mlp_ratio)
        channel_mlp_dim = round(dim * channel_mlp_ratio)
        self.drop_path = DropPath(drop_path
            ) if drop_path > 0.0 else nn.Identity()
        self.token_mix = VanillaMlp(num_patch, token_mlp_dim, num_patch,
            act_layer, drop)
        self.channel_mix = VanillaMlp(dim, channel_mlp_dim, dim, act_layer,
            drop)

    def forward(self, x):
        y = self.norm1(x).transpose(1, 2)
        y = self.drop_path(self.token_mix(y)).transpose(1, 2)
        x = x + y
        y = self.norm2(x)
        x = x + self.drop_path(self.channel_mix(y))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_patch': 4, 'dim': 4, 'token_mlp_ratio': 4,
        'channel_mlp_ratio': 4}]
