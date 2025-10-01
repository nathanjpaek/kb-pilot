import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
import torch.onnx


class Highway(nn.Module):

    def __init__(self, e_word):
        super(Highway, self).__init__()
        self.embed_size = e_word
        self.w_proj = nn.Linear(self.embed_size, self.embed_size, bias=True)
        self.w_gate = nn.Linear(self.embed_size, self.embed_size, bias=True)

    def forward(self, x_convout: 'torch.Tensor') ->torch.Tensor:
        x_proj = F.relu(self.w_proj(x_convout))
        sig = nn.Sigmoid()
        x_gate = sig(self.w_gate(x_convout))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_convout
        return x_highway


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'e_word': 4}]
