import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, target, mask):
        x = x.contiguous().view(-1, x.size(2))
        target = target.contiguous().view(-1, 1)
        mask = mask.contiguous().view(-1, 1)
        output = -x.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


def get_inputs():
    return [torch.ones([256, 4, 4], dtype=torch.int64), torch.ones([256],
        dtype=torch.int64), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
