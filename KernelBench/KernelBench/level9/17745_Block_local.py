import math
import torch
import numpy as np
from torch import nn
from torch.nn.modules.utils import _pair
from functools import partial
import torch.utils.data
import torch.nn.parallel
from torch import optim as optim


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


class DCNv2(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, dilation=1, deformable_groups=1):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels,
            *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        assert 2 * self.deformable_groups * self.kernel_size[0
            ] * self.kernel_size[1] == offset.shape[1]
        assert self.deformable_groups * self.kernel_size[0] * self.kernel_size[
            1] == mask.shape[1]
        return dcn_v2_conv(input, offset, mask, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.deformable_groups)


class DCN(DCNv2):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, dilation=1, deformable_groups=1):
        super(DCN, self).__init__(in_channels, out_channels, kernel_size,
            stride, padding, dilation, deformable_groups)
        channels_ = self.deformable_groups * 3 * self.kernel_size[0
            ] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels, channels_,
            kernel_size=self.kernel_size, stride=self.stride, padding=self.
            padding, bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return dcn_v2_conv(input, offset, mask, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.deformable_groups)


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


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
        attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads
            ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LocalAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
        attn_drop=0.0, proj_drop=0.0, local_ks=3, length=196):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        mask = torch.ones(length, length)
        for i in range(length):
            for j in range(i - local_ks // 2, i + local_ks // 2 + 1, 1):
                j = min(max(0, j), length - 1)
                mask[i, j] = 0
        mask = mask.unsqueeze(0).unsqueeze(1)
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads
            ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.masked_fill_(self.mask.bool(), -np.inf)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LocalBranch(nn.Module):

    def __init__(self, dim, local_type='conv', local_ks=3, length=196,
        num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0,
        proj_drop=0.0):
        super().__init__()
        self.local_type = local_type
        if local_type == 'conv':
            self.linear = nn.Linear(dim, dim)
            self.local = nn.Conv1d(dim, dim, kernel_size=local_ks, padding=
                local_ks // 2, padding_mode='zeros', groups=1)
        elif local_type == 'dcn':
            self.linear = nn.Linear(dim, dim)
            self.local = DCN(dim, dim, kernel_size=(local_ks, 1), stride=1,
                padding=(local_ks // 2, 0), deformable_groups=2)
        elif local_type == 'attn':
            self.local = LocalAttention(dim, num_heads=num_heads, qkv_bias=
                qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop
                =proj_drop, local_ks=local_ks, length=length)
        else:
            self.local = nn.Identity()

    def forward(self, x):
        if self.local_type in ['conv']:
            x = self.linear(x)
            x = x.permute(0, 2, 1)
            x = self.local(x)
            x = x.permute(0, 2, 1)
            return x
        elif self.local_type == 'dcn':
            x = self.linear(x)
            x = x.permute(0, 2, 1).unsqueeze(3).contiguous()
            x = self.local(x)
            x = x.squeeze(3).permute(0, 2, 1)
            return x
        elif self.local_type == 'attn':
            x = self.local(x)
            return x
        else:
            x = self.local(x)
            return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block_local(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
        qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn
        .GELU, norm_layer=partial(nn.LayerNorm, eps=1e-06), local_type=
        'conv', local_ks=3, length=196, local_ratio=0.5, ffn_type='base'):
        super().__init__()
        local_dim = int(dim * local_ratio)
        self.global_dim = dim - local_dim
        div = 2
        self.num_heads = num_heads // div
        self.norm1 = norm_layer(self.global_dim)
        self.norm1_local = norm_layer(local_dim)
        self.attn = Attention(self.global_dim, num_heads=self.num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop)
        self.local = LocalBranch(local_dim, local_type=local_type, local_ks
            =local_ks, length=length, num_heads=self.num_heads, qkv_bias=
            qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path
            ) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if ffn_type == 'base':
            MLP = Mlp
        else:
            raise Exception('invalid ffn_type: {}'.format(ffn_type))
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop)

    def forward(self, x):
        x_attn = self.drop_path(self.attn(self.norm1(x[:, :, :self.
            global_dim])))
        x_local = self.drop_path(self.local(self.norm1_local(x[:, :, self.
            global_dim:])))
        x = x + torch.cat([x_attn, x_local], dim=2)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'num_heads': 4}]
