import torch
import torch.nn as nn


class Norm(nn.Module):

    def __init__(self, d_model, eps=1e-06):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_variance = x.std(dim=-1, keepdim=True)
        normalized_x = (x - x_mean) / (x_variance + self.eps)
        y = self.alpha * normalized_x + self.bias
        return y


class AddNorm(nn.Module):

    def __init__(self, d_model, dropout_pct=0.1):
        super().__init__()
        self.norm = Norm(d_model)
        self.dropout = nn.Dropout(dropout_pct)

    def forward(self, x, attn_output):
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4}]
