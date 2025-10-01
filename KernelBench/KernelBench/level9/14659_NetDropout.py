import torch
from torch import nn
from torch.nn import functional as F


class NetDropout(nn.Module):

    def __init__(self, nclasses, img, nchans1=10, dropout_prob=0.4):
        super().__init__()
        nchannels, _nrows, _ncols = img.shape
        self.conv1 = nn.Conv2d(nchannels, nchans1, kernel_size=3, padding=1)
        self.conv1_dropout = nn.Dropout2d(dropout_prob)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = self.conv1_dropout(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nclasses': 4, 'img': torch.rand([4, 4, 4])}]
