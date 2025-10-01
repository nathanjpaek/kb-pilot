import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn as nn
import torch.cuda


class SpatialAttention(nn.Module):

    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        feat_avg = torch.mean(x, dim=1, keepdim=True)
        feat_max = torch.max(x, dim=1, keepdim=True)[0]
        feature = torch.cat((feat_avg, feat_max), dim=1)
        return self.sigmoid(self.conv(feature))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
