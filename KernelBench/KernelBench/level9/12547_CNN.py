import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, embed_size, hidden_size):
        super(CNN, self).__init__()
        self.hidden_size = hidden_size
        self.conv2d = nn.Conv2d(embed_size, hidden_size, (1, 5), bias=True)

    def forward(self, x):
        x = self.conv2d(x)
        x = torch.max(F.relu(x), dim=-1)[0]
        x = x.permute(0, 2, 1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'embed_size': 4, 'hidden_size': 4}]
