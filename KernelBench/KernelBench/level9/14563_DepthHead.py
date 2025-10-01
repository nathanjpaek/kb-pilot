import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthHead(nn.Module):

    def __init__(self, input_dim=256, hidden_dim=128, scale=False):
        super(DepthHead, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_d, act_fn=F.tanh):
        out = self.conv2(self.relu(self.conv1(x_d)))
        return act_fn(out)


def get_inputs():
    return [torch.rand([4, 256, 64, 64])]


def get_init_inputs():
    return [[], {}]
