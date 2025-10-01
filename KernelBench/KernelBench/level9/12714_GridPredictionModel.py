import torch
import torch.nn as nn
import torch.nn.functional as F


class GridPredictionModel(nn.Module):

    def __init__(self):
        super(GridPredictionModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size
            =3, padding=2)
        self.conv2 = nn.Conv2d(in_channels=100, out_channels=1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x[:, :, 1:-1, 1:-1]


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
