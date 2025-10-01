import torch
import torch.nn as nn
import torch.nn.init as init


class LayerNorm(nn.Module):

    def __init__(self, d_hid, eps=1e-06):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid))
        self.beta = nn.Parameter(torch.zeros(d_hid))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        ln_out = (x - mean) / (std + self.eps)
        ln_out = self.gamma * ln_out + self.beta
        return ln_out


class Linear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)
        init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)


class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = LayerNorm(d_model)

    def forward(self, x):
        residual = x
        output = self.relu(self.fc1(x))
        output = self.dropout(self.fc2(output))
        output = self.layernorm(output + residual)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4}]
