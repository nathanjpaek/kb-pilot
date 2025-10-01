import torch
import torch.nn as nn


class DropBlockT_1d(nn.Module):

    def __init__(self, keep_prob=0.9):
        super(DropBlockT_1d, self).__init__()
        self.keep_prob = keep_prob

    def forward(self, input, mask):
        n, c, t, v = input.size()
        input1 = input.permute(0, 1, 3, 2).contiguous().view(n, c * v, t)
        mask[mask >= self.keep_prob] = 2.0
        mask[mask < self.keep_prob] = 1.0
        mask[mask == 2.0] = 0.0
        return (input1 * mask * mask.numel() / mask.sum()).view(n, c, v, t
            ).permute(0, 1, 3, 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 16, 4])]


def get_init_inputs():
    return [[], {}]
