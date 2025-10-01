import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


class JaccardLoss(_Loss):

    def __init__(self):
        super(JaccardLoss, self).__init__()

    def forward(self, output, target):
        output = F.sigmoid(output)
        intersection = torch.sum(output * target)
        union = torch.sum(output) + torch.sum(target)
        jac = intersection / (union - intersection + 1e-07)
        return 1 - jac


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
