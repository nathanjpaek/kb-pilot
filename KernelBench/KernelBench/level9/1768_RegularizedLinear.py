import torch
import torch.nn as nn


class RegularizedLinear(nn.Linear):

    def __init__(self, *args, ar_weight=0.001, l1_weight=0.001, **kwargs):
        super(RegularizedLinear, self).__init__(*args, **kwargs)
        self.ar_weight = ar_weight
        self.l1_weight = l1_weight
        self._losses = {}

    def forward(self, input):
        output = super(RegularizedLinear, self).forward(input)
        self._losses['activity_regularization'] = (output * output).sum(
            ) * self.ar_weight
        self._losses['l1_weight_regularization'] = torch.abs(self.weight).sum(
            ) * self.l1_weight
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
