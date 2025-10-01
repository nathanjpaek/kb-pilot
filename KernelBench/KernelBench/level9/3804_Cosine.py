from _paritybench_helpers import _mock_config
import torch
from torch.optim.lr_scheduler import *


class Cosine(torch.nn.Module):

    def __init__(self, config):
        super().__init__()

    def forward(self, src, tgt):
        src = src.float()
        tgt = tgt.float()
        return (torch.matmul(src, tgt.transpose(2, 1)) / (src.norm(p=2, dim
            =-1, keepdim=True) * tgt.norm(p=2, dim=-1, keepdim=True) + 1e-09)
            ).squeeze()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config()}]
