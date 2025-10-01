import torch
import torch.nn as nn
import torch.hub


class MaskLoss(nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MaskLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        N, _W = input.size()
        torch.min(input, target)
        values, index = torch.max(target, 0)
        1 / (1 + torch.exp(-100 * (target - 0.55 * values)))
        sums = []
        for n in range(N):
            value = values[n]
            index[n]
            tar = target[n]
            inp = input[n]
            a = torch.min(inp, tar)
            b = 1 / (1 + torch.exp(-100 * (tar - 0.55 * value)))
            sums.append(2 * torch.div(torch.dot(a, b), torch.sum(inp +
                target, axis=-1)))
        sums = torch.stack(sums)
        sums[torch.isnan(sums)] = 0.0
        return sums.mean()


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
