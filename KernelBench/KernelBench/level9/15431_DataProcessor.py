import torch
import torch.nn as nn


class DataProcessor(nn.Module):

    def __init__(self):
        super(DataProcessor, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        x = self.pool(x)
        x = torch.squeeze(x)
        x = x.permute(1, 2, 0)
        return x.view(-1, x.size(-1))


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
