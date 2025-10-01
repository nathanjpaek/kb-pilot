import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class DECLoss(nn.Module):

    def __init__(self):
        super(DECLoss, self).__init__()

    def target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        return Variable((weight.t() / weight.sum(1)).t().data,
            requires_grad=True)

    def KL_div(self, q, p):
        res = torch.sum(p * torch.log(p / q))
        return res

    def forward(self, q):
        p = self.target_distribution(q)
        return self.KL_div(q, p)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
