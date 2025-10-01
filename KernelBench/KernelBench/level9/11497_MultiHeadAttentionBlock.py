import math
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn


def mask_logits(inputs, mask, mask_value=-1e+30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value


class Conv1D(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0,
        bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim,
            kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        return x.transpose(1, 2)


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, drop_rate):
        super(MultiHeadAttentionBlock, self).__init__()
        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (
            dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(dim / num_heads
            ), num_heads, dim
        self.dropout = nn.Dropout(p=drop_rate)
        self.query = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=
            1, padding=0, bias=True)
        self.key = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1,
            padding=0, bias=True)
        self.value = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=
            1, padding=0, bias=True)
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-06)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-06)
        self.out_layer = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1,
            stride=1, padding=0, bias=True)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)

    def forward(self, x, mask=None):
        output = self.layer_norm1(x)
        output = self.dropout(output)
        query = self.transpose_for_scores(self.query(output))
        key = self.transpose_for_scores(self.key(output))
        value = self.transpose_for_scores(self.value(output))
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_size)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attention_scores = mask_logits(attention_scores, mask)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        value = torch.matmul(attention_probs, value)
        value = self.combine_last_two_dim(value.permute(0, 2, 1, 3))
        output = self.dropout(value)
        residual = output + x
        output = self.layer_norm2(residual)
        output = self.dropout(output)
        output = self.out_layer(output)
        output = self.dropout(output) + residual
        return output


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'num_heads': 4, 'drop_rate': 0.5}]
