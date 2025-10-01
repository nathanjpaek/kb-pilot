import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


class SmoothJaccardLoss(_Loss):

    def __init__(self, smooth=100):
        super(SmoothJaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, output, target):
        output = F.sigmoid(output)
        target = target.float()
        intersection = torch.sum(output * target)
        union = torch.sum(output) + torch.sum(target)
        jac = (intersection + self.smooth) / (union - intersection + self.
            smooth)
        return 1 - jac


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
