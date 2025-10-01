import torch
import torch.optim.lr_scheduler
import torch.utils.data
from torchvision.transforms import *


class DiceLoss(torch.nn.Module):

    def init(self):
        super(DiceLoss, self).init()

    def forward(self, pred, target):
        smooth = 1.0
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - (2.0 * intersection + smooth) / (A_sum + B_sum + smooth)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
