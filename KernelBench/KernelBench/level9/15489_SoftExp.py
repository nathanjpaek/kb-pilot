import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftExp(nn.Module):

    def __init__(self, input_size):
        super(SoftExp, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor(input_size))

    def forward(self, data):
        self.alpha.data.clamp_(-1, 1)
        positives = torch.gt(F.threshold(self.alpha, 0, 0), 0)
        negatives = torch.gt(F.threshold(-self.alpha, 0, 0), 0)
        output = data.clone()
        pos_out = (torch.exp(self.alpha * data) - 1) / self.alpha + self.alpha
        neg_out = -torch.log(1 - self.alpha * (data + self.alpha)) / self.alpha
        output.masked_scatter_(positives, pos_out.masked_select(positives))
        output.masked_scatter_(negatives, neg_out.masked_select(negatives))
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
