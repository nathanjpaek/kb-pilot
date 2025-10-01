import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, hidden):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden, 1, bias=False)

    def forward(self, x, mask=None):
        weights = self.linear(x)
        if mask is not None:
            weights = weights.mask_fill(mask.unsqueeze(2) == 0, float('-inf'))
        alpha = torch.softmax(weights, dim=1)
        outputs = x * alpha
        outputs = torch.sum(outputs, dim=1)
        return outputs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden': 4}]
