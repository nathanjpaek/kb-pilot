import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.nn import functional as F


class Similarity(nn.Module):

    def __init__(self, mem_dim, hidden_dim, num_classes):
        super(Similarity, self).__init__()
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.wh = nn.Linear(2 * self.mem_dim, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, lvec, rvec):
        mult_dist = torch.mul(lvec, rvec)
        abs_dist = torch.abs(torch.add(lvec, -rvec))
        vec_dist = torch.cat((mult_dist, abs_dist), 1)
        out = F.sigmoid(self.wh(vec_dist))
        out = F.log_softmax(self.wp(out), dim=1)
        return out


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'mem_dim': 4, 'hidden_dim': 4, 'num_classes': 4}]
