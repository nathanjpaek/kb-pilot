import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.hub


class BerhuLoss(nn.Module):

    def __init__(self):
        super(BerhuLoss, self).__init__()
        self.name = 'Berhu'

    def forward(self, input, target, mask=None):
        assert input.shape == target.shape
        if mask is not None:
            input = input[mask]
            target = target[mask]
        diff = torch.abs(input - target)
        c = 0.2 * torch.max(diff)
        diff_square = (torch.square(diff) + torch.square(c)) / (2 * c)
        diff_square[diff <= c] = 0
        diff_copy = diff.clone()
        diff_copy[diff_copy > c] = 0
        diff_copy += diff_square
        loss = torch.mean(diff_copy)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
