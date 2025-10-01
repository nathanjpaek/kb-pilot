import torch
import torch.nn as nn


class AdaIn(nn.Module):

    def __init__(self):
        super(AdaIn, self).__init__()
        self.eps = 1e-05

    def forward(self, x, mean_style, std_style):
        B, C, H, W = x.shape
        feature = x.view(B, C, -1)
        std_feat = (torch.std(feature, dim=2) + self.eps).view(B, C, 1)
        mean_feat = torch.mean(feature, dim=2).view(B, C, 1)
        adain = std_style * (feature - mean_feat) / std_feat + mean_style
        adain = adain.view(B, C, H, W)
        return adain


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 16]), torch.rand([4,
        4, 16])]


def get_init_inputs():
    return [[], {}]
