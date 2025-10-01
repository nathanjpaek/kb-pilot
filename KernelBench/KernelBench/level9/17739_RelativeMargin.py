import torch
import torch.nn as nn


class RelativeMargin(nn.Module):

    def __init__(self):
        super(RelativeMargin, self).__init__()

    def forward(self, x1, x2, y1, y2, t, reduce=True):
        if reduce:
            loss = torch.mean(torch.clamp(torch.abs(y1 - y2) - t * (x1 - x2
                ), 0.0))
        else:
            loss = torch.sum(torch.clamp(torch.abs(y1 - y2) - t * (x1 - x2),
                0.0))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
