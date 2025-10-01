import torch
from torch import nn
import torch.nn.parallel
import torch.utils.data


class LoRALayer(nn.Module):

    def __init__(self, n_in, n_out=None, adapter_dim=16, adapter_alpha=32):
        super(LoRALayer, self).__init__()
        if not n_out:
            n_out = n_in
        self.adapter_dim = adapter_dim
        self.adapter_alpha = adapter_alpha
        self.adapter_proj_1 = nn.Linear(n_in, adapter_dim, bias=False)
        nn.init.normal_(self.adapter_proj_1.weight, std=0.02)
        self.adapter_proj_2 = nn.Linear(adapter_dim, n_out, bias=False)
        self.adapter_proj_2.weight.data.zero_()

    def forward(self, x):
        scale_factor = self.adapter_dim / self.adapter_alpha
        result = torch.matmul(x, self.adapter_proj_1.weight.type_as(x).T)
        return torch.matmul(result, self.adapter_proj_2.weight.type_as(x).T
            ) * scale_factor


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_in': 4}]
