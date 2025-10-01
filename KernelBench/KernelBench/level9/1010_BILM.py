import torch
import torch.nn as nn


class BILM(nn.Module):

    def __init__(self):
        super(BILM, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, feat):
        pos_sig = torch.sigmoid(feat)
        neg_sig = -1 * pos_sig
        pos_sig = self.maxpool1(pos_sig)
        neg_sig = self.maxpool2(neg_sig)
        sum_sig = pos_sig + neg_sig
        x = feat * sum_sig
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
