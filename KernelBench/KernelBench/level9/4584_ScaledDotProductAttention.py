import math
import torch
from torch.nn import functional as F
from torch import nn
from torchvision import models as models
import torch.onnx
import torch.nn


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dropout_ratio=0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, query, keys, values, mask=None):
        attn = torch.matmul(query, keys.transpose(-2, -1))
        attn /= math.sqrt(query.shape[-1])
        if mask is None:
            mask = attn.new_ones(attn.shape)
        if mask.dim() < attn.dim():
            mask = mask.unsqueeze(-2)
        mask = self.dropout(mask)
        attn = attn.masked_fill(mask == 0, -1000.0)
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, values)
        return output, attn


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
