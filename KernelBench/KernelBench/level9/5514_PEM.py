from _paritybench_helpers import _mock_config
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
import torch.nn.parallel


class PEM(torch.nn.Module):

    def __init__(self, opt):
        super(PEM, self).__init__()
        self.feat_dim = opt['pem_feat_dim']
        self.batch_size = opt['pem_batch_size']
        self.hidden_dim = opt['pem_hidden_dim']
        self.u_ratio_m = opt['pem_u_ratio_m']
        self.u_ratio_l = opt['pem_u_ratio_l']
        self.output_dim = 1
        self.pem_best_loss = 1000000
        self.fc1 = torch.nn.Linear(in_features=self.feat_dim, out_features=
            self.hidden_dim, bias=True)
        self.fc2 = torch.nn.Linear(in_features=self.hidden_dim,
            out_features=self.output_dim, bias=True)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x = F.relu(0.1 * self.fc1(x))
        x = torch.sigmoid(0.1 * self.fc2(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'opt': _mock_config(pem_feat_dim=4, pem_batch_size=4,
        pem_hidden_dim=4, pem_u_ratio_m=4, pem_u_ratio_l=4)}]
