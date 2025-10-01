import torch
import torch.nn as nn
import torch.nn.functional as F


class Similarity(nn.Module):

    def __init__(self, cuda, mem_dim, hidden_dim, num_classes):
        super(Similarity, self).__init__()
        self.cudaFlag = cuda
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.wh = nn.Linear(2 * self.mem_dim, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, lvec, rvec):
        mult_dist = F.torch.mul(lvec, rvec)
        abs_dist = F.torch.abs(F.torch.add(lvec, -rvec))
        vec_dist = F.torch.cat((mult_dist, abs_dist), 1)
        out = F.sigmoid(self.wh(vec_dist))
        out = F.log_softmax(self.wp(out))
        return out


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'cuda': False, 'mem_dim': 4, 'hidden_dim': 4,
        'num_classes': 4}]
