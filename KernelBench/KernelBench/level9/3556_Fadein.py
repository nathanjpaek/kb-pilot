from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
import torch.utils.data


class Fadein(nn.Module):

    def __init__(self, cfg):
        super(Fadein, self).__init__()
        self.alpha = 0.0

    def update_alpha(self, delta):
        self.alpha = self.alpha + delta
        self.alpha = max(0, min(self.alpha, 1.0))

    def forward(self, x):
        return torch.add(x[0].mul(1.0 - self.alpha), x[1].mul(self.alpha))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'cfg': _mock_config()}]
