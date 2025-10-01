import torch
import torch.nn as nn


class DropBlock_Ske(nn.Module):

    def __init__(self, num_point=25, keep_prob=0.9):
        super(DropBlock_Ske, self).__init__()
        self.keep_prob = keep_prob
        self.num_point = num_point

    def forward(self, input, mask):
        n, _c, _t, _v = input.size()
        mask[mask >= self.keep_prob] = 2.0
        mask[mask < self.keep_prob] = 1.0
        mask[mask == 2.0] = 0.0
        mask = mask.view(n, 1, 1, self.num_point)
        return input * mask * mask.numel() / mask.sum()


def get_inputs():
    return [torch.rand([4, 1, 1, 25]), torch.rand([4, 1, 1, 25])]


def get_init_inputs():
    return [[], {}]
