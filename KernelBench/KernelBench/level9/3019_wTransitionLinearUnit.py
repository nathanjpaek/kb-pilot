from torch.nn import Module
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from scipy.sparse import *


class wTransitionLinearUnit(Module):

    def __init__(self, ori_dim, tar_dim):
        super(wTransitionLinearUnit, self).__init__()
        self.linear_1 = torch.nn.Linear(tar_dim, ori_dim)
        self.linear_2 = torch.nn.Linear(tar_dim, tar_dim)
        self.linear_3 = torch.nn.Linear(tar_dim, tar_dim)
        self.ori_dim = ori_dim
        self.tar_dim = tar_dim
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, last_w, z_cov):
        z_cov = (z_cov - z_cov.mean()) / z_cov.std()
        hidden = F.relu(self.linear_1(z_cov.t()))
        w_update = self.linear_2(hidden)
        update_gate = torch.sigmoid(self.linear_3(hidden))
        w_update = torch.clamp(w_update, min=-0.1, max=0.1)
        return (1 - update_gate) * last_w + w_update * update_gate


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'ori_dim': 4, 'tar_dim': 4}]
