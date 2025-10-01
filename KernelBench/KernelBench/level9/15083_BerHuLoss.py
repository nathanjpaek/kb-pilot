import torch
import torch.nn as nn


class BerHuLoss(nn.Module):

    def __init__(self, scale=0.5, eps=1e-05):
        super(BerHuLoss, self).__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, pred, gt):
        img1 = torch.zeros_like(pred)
        img2 = torch.zeros_like(gt)
        img1 = img1.copy_(pred)
        img2 = img2.copy_(gt)
        img1 = img1[img2 > self.eps]
        img2 = img2[img2 > self.eps]
        diff = torch.abs(img1 - img2)
        threshold = self.scale * torch.max(diff).detach()
        mask = diff > threshold
        diff[mask] = ((img1[mask] - img2[mask]) ** 2 + threshold ** 2) / (2 *
            threshold + self.eps)
        return diff.sum() / diff.numel()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
