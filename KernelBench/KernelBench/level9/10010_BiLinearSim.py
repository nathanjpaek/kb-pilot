from _paritybench_helpers import _mock_config
import torch
from torch.optim.lr_scheduler import *


class BiLinearSim(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.linear = torch.nn.Linear(config.hidden_size, config.
            hidden_size, bias=False)

    def forward(self, src, tgt):
        src_ = self.linear(src)
        output = torch.matmul(src_, tgt.transpose(2, 1))
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4)}]
