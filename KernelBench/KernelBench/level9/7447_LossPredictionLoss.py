import torch
import torch.nn as nn


class LossPredictionLoss(nn.Module):

    def __init__(self, margin=1.0):
        super(LossPredictionLoss, self).__init__()
        self.margin = margin

    def forward(self, input, target):
        input = (input - input.flip(0))[:len(input) // 2]
        target = (target - target.flip(0))[:len(target) // 2]
        target = target.detach()
        one = 2 * torch.sign(torch.clamp(target, min=0)) - 1
        loss = torch.sum(torch.clamp(self.margin - one * input, min=0))
        loss = loss / input.size(0)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
