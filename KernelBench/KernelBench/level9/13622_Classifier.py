import torch
import torch.nn as nn


class Classifier(nn.Module):

    def __init__(self, in_dim, num_classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(in_dim, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'num_classes': 4}]
