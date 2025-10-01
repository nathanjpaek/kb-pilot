import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


class JaccardScore(_Loss):

    def __init__(self):
        super(JaccardScore, self).__init__()

    def forward(self, output, target):
        output = F.sigmoid(output)
        target = target.float()
        intersection = (output * target).sum()
        union = output.sum() + target.sum()
        jac = intersection / (union - intersection + 1e-07)
        return jac

    def __str__(self):
        return 'JaccardScore'


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
