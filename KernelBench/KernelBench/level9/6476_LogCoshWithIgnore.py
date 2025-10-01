import torch
import torch.nn.functional
from torch import nn


class LogCoshWithIgnore(nn.Module):

    def __init__(self, ignore_value, fraction: 'float'=1.0):
        super().__init__()
        self.ignore_value = ignore_value
        self.fraction = fraction

    def forward(self, output, target):
        r = output - target
        log2 = 0.30102999566
        loss = torch.logaddexp(r, -r) - log2
        loss = torch.masked_fill(loss, target.eq(self.ignore_value), 0)
        if self.fraction < 1:
            loss = loss.reshape(loss.size(0), -1)
            M = loss.size(1)
            num_elements_to_keep = int(M * self.fraction)
            loss, _ = torch.topk(loss, k=num_elements_to_keep, dim=1,
                largest=False, sorted=False)
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'ignore_value': 4}]
