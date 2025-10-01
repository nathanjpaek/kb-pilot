import torch
import torch.nn as nn


class DepthL1Loss(nn.Module):

    def __init__(self, eps=1e-05):
        super(DepthL1Loss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt):
        bs = pred.size()[0]
        img1 = torch.zeros_like(pred)
        img2 = torch.zeros_like(gt)
        img1 = img1.copy_(pred)
        img2 = img2.copy_(gt)
        mask = gt > self.eps
        img1[~mask] = 0.0
        img2[~mask] = 0.0
        loss = nn.L1Loss(reduction='sum')(img1, img2)
        loss = loss / mask.float().sum() * bs
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
