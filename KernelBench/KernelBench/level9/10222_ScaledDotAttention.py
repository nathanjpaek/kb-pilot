import torch
import torch.nn as nn
from torch.nn import LayerNorm


def scaled_dot_attention(q, k, v, mask=None, noise=0, dropout=lambda x: x):
    """
    :param q: queries, (batch, time1, channels1)
    :param k: keys, (batch, time2, channels1)
    :param v: values, (batch, time2, channels2)
    :param mask: boolean mask, (batch, time1, time2)
    :param dropout: a dropout function - this allows keeping dropout as a module -> better control when training/eval
    :return: (batch, time1, channels2), (batch, time1, time2)
    """
    weights = torch.matmul(q, k.transpose(2, 1))
    if mask is not None:
        weights = weights.masked_fill(~mask, float('-inf'))
    if noise:
        weights += noise * torch.randn(weights.shape)
    weights = torch.softmax(weights, dim=-1)
    weights = dropout(weights)
    result = torch.matmul(weights, v)
    return result, weights


def mask(x, lengths, dim=-1):
    assert dim != 0, 'Masking not available for batch dimension'
    assert len(lengths) == x.shape[0
        ], 'Lengths must contain as many elements as there are items in the batch'
    lengths = torch.as_tensor(lengths)
    to_expand = [1] * (x.ndim - 1) + [-1]
    mask = torch.arange(x.shape[dim]).expand(to_expand).transpose(dim, -1
        ).expand(x.shape)
    mask = mask < lengths.expand(to_expand).transpose(0, -1)
    return mask


class Conv1d(nn.Conv1d):
    """A wrapper around nn.Conv1d, that works on (batch, time, channels)"""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        dilation=1, groups=1, bias=True, padding=0):
        super(Conv1d, self).__init__(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, stride=stride, dilation=
            dilation, groups=groups, bias=bias, padding=padding)

    def forward(self, x):
        return super().forward(x.transpose(2, 1)).transpose(2, 1)


class ScaledDotAttention(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, noise=0,
        normalize=False, dropout=False):
        super(ScaledDotAttention, self).__init__()
        self.noise = noise
        self.dropout = torch.nn.Dropout(p=dropout)
        self.normalize = normalize
        self.fc_query = Conv1d(in_channels, hidden_channels)
        self.fc_keys = Conv1d(in_channels, hidden_channels)
        if normalize:
            self.qnorm = LayerNorm(in_channels)
            self.knorm = LayerNorm(in_channels)
        self.fc_keys.weight = torch.nn.Parameter(self.fc_query.weight.clone())
        self.fc_keys.bias = torch.nn.Parameter(self.fc_query.bias.clone())
        self.fc_values = Conv1d(in_channels, hidden_channels)
        self.fc_out = Conv1d(hidden_channels, out_channels)

    def forward(self, q, k, v, mask=None):
        """
        :param q: queries, (batch, time1, channels1)
        :param k: keys, (batch, time2, channels1)
        :param v: values, (batch, time2, channels2)
        :param mask: boolean mask, (batch, time1, time2)
        :return: (batch, time1, channels2), (batch, time1, time2)
        """
        noise = self.noise if self.training else 0
        if self.normalize:
            q = self.qnorm(q)
            k = self.knorm(k)
        alignment, weights = scaled_dot_attention(self.fc_query(q), self.
            fc_keys(k), self.fc_values(v), mask, noise=noise, dropout=self.
            dropout)
        alignment = self.fc_out(alignment)
        return alignment, weights


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'in_channels': 4, 'hidden_channels': 4, 'out_channels': 4}]
