import torch
from torch import nn


class dce_loss(nn.Module):

    def __init__(self, n_classes, feat_dim, init_weight=True):
        super(dce_loss, self).__init__()
        self.n_classes = n_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.feat_dim, self.
            n_classes), requires_grad=True)
        nn.init.kaiming_normal_(self.centers)

    def forward(self, x):
        features_square = torch.sum(torch.pow(x, 2), 1, keepdim=True)
        centers_square = torch.sum(torch.pow(self.centers, 2), 0, keepdim=True)
        features_into_centers = 2 * torch.matmul(x, self.centers)
        dist = features_square + centers_square - features_into_centers
        return self.centers, -dist


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_classes': 4, 'feat_dim': 4}]
