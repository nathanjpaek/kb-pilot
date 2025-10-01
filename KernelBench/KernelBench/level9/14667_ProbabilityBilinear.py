import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_prob(a, dim=-1):
    """Perform 1-norm along the specific dimension."""
    return a / a.sum(dim=dim, keepdim=True)


class ProbabilityBilinear(nn.Bilinear):

    def __init__(self, in1_features, in2_features, out_features, bias=False,
        norm=True):
        assert bias is False, 'Bias regularization for SOFTMAX is not implemented.'
        super().__init__(in1_features, in2_features, out_features, bias)
        self.norm = norm

    def forward(self, input1, input2):
        weight = self._regulize_parameter(self.weight)
        output = F.bilinear(input1, input2, weight, None)
        if self.norm:
            output = normalize_prob(output)
        return output

    def _regulize_parameter(self, p):
        return F.softmax(p, dim=0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in1_features': 4, 'in2_features': 4, 'out_features': 4}]
