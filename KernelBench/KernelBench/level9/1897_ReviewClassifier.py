import torch
import torch.nn.functional as F
import torch.nn as nn


class ReviewClassifier(nn.Module):

    def __init__(self, n_feature):
        super(ReviewClassifier, self).__init__()
        self.lf = nn.Linear(n_feature, 1, dtype=torch.float32)

    def forward(self, x):
        out = self.lf(x)
        out = F.sigmoid(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_feature': 4}]
