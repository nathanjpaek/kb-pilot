import torch
from torch import nn


class ResNetClassifier(nn.Module):

    def __init__(self, n_class, len_feature):
        super().__init__()
        self.len_feature = len_feature
        self.classifier = nn.Linear(self.len_feature, n_class)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        x = x.mean(dim=-1)
        x = self.classifier(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_class': 4, 'len_feature': 4}]
