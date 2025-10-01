import math
import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """

    def __init__(self, num_channels, eps=1e-05, affine=True, device=None,
        dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones([1, num_channels, 1], **
                factory_kwargs))
            self.bias = nn.Parameter(torch.zeros([1, num_channels, 1], **
                factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x ** 2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)
        if self.affine:
            out *= self.weight
            out += self.bias
        return out


class MaskedConv1D(nn.Module):
    """
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only support a sub set of 1d convs
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        assert kernel_size % 2 == 1 and kernel_size // 2 == padding
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias, padding_mode)
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x, mask):
        _B, _C, T = x.size()
        assert T % self.stride == 0
        out_conv = self.conv(x)
        if self.stride > 1:
            out_mask = F.interpolate(mask.float(), size=T // self.stride,
                mode='nearest')
        else:
            out_mask = mask.float()
        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        return out_conv, out_mask


class MaskedMHCA(nn.Module):
    """
    Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(self, n_embd, n_head, n_qx_stride=1, n_kv_stride=1,
        attn_pdrop=0.0, proj_pdrop=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)
        assert n_qx_stride == 1 or n_qx_stride % 2 == 0
        assert n_kv_stride == 1 or n_kv_stride % 2 == 0
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = MaskedConv1D(self.n_embd, self.n_embd,
            kernel_size, stride=stride, padding=padding, groups=self.n_embd,
            bias=False)
        self.query_norm = LayerNorm(self.n_embd)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = MaskedConv1D(self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False)
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(self.n_embd, self.n_embd,
            kernel_size, stride=stride, padding=padding, groups=self.n_embd,
            bias=False)
        self.value_norm = LayerNorm(self.n_embd)
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask):
        B, C, _T = x.size()
        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q)
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v)
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        att = q * self.scale @ k.transpose(-2, -1)
        att = att.masked_fill(torch.logical_not(kv_mask[:, :, None, :]),
            float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        out = att @ (v * kv_mask[:, :, :, None].float())
        out = out.transpose(2, 3).contiguous().view(B, C, -1)
        out = self.proj_drop(self.proj(out)) * qx_mask.float()
        return out, qx_mask


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'n_embd': 4, 'n_head': 4}]
