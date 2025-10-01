from _paritybench_helpers import _mock_config
import torch
from torch import nn
import torch.nn.init


class Fusion(nn.Module):

    def __init__(self, opt):
        super(Fusion, self).__init__()
        self.f_size = opt.embed_size
        self.gate0 = nn.Linear(self.f_size, self.f_size)
        self.gate1 = nn.Linear(self.f_size, self.f_size)
        self.fusion0 = nn.Linear(self.f_size, self.f_size)
        self.fusion1 = nn.Linear(self.f_size, self.f_size)

    def forward(self, vec1, vec2):
        features_1 = self.gate0(vec1)
        features_2 = self.gate1(vec2)
        t = torch.sigmoid(self.fusion0(features_1) + self.fusion1(features_2))
        f = t * features_1 + (1 - t) * features_2
        return f


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'opt': _mock_config(embed_size=4)}]
