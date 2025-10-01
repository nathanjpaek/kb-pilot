import torch
import torch.nn as nn


class Mlp(nn.Module):

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


class Mix(nn.Module):

    def __init__(self, dim, y_dim, num_heads, mlp_ratio=4.0, y_mlp_ratio=
        4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
        drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(y_dim)
        y_mlp_hidden_dim = int(y_dim / y_mlp_ratio)
        self.channel_mlp = Mlp(in_features=y_dim, hidden_features=
            y_mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path
            ) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.channel_mlp(self.norm1(x.transpose(1, 2
            ).contiguous())).transpose(1, 2).contiguous())
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'y_dim': 4, 'num_heads': 4}]
