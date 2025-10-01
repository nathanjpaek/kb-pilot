from _paritybench_helpers import _mock_config
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
import torch.nn.parallel


class TEM(torch.nn.Module):

    def __init__(self, opt):
        super(TEM, self).__init__()
        self.feat_dim = opt['tem_feat_dim']
        self.temporal_dim = opt['temporal_scale']
        self.batch_size = opt['tem_batch_size']
        self.c_hidden = opt['tem_hidden_dim']
        self.tem_best_loss = 10000000
        self.output_dim = 3
        self.conv1 = torch.nn.Conv1d(in_channels=self.feat_dim,
            out_channels=self.c_hidden, kernel_size=3, stride=1, padding=1,
            groups=1)
        self.conv2 = torch.nn.Conv1d(in_channels=self.c_hidden,
            out_channels=self.c_hidden, kernel_size=3, stride=1, padding=1,
            groups=1)
        self.conv3 = torch.nn.Conv1d(in_channels=self.c_hidden,
            out_channels=self.output_dim, kernel_size=1, stride=1, padding=0)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(0.01 * self.conv3(x))
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'opt': _mock_config(tem_feat_dim=4, temporal_scale=1.0,
        tem_batch_size=4, tem_hidden_dim=4)}]
