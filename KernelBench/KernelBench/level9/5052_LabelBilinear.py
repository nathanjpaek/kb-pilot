import torch
from torch import nn
import torch.utils.data


class LabelBilinear(nn.Module):
    """helper module for Biaffine Dependency Parser predicting label
    """

    def __init__(self, in1_features, in2_features, num_label, bias=True):
        super(LabelBilinear, self).__init__()
        self.bilinear = nn.Bilinear(in1_features, in2_features, num_label,
            bias=bias)
        self.lin = nn.Linear(in1_features + in2_features, num_label, bias=False
            )

    def forward(self, x1, x2):
        output = self.bilinear(x1, x2)
        output += self.lin(torch.cat([x1, x2], dim=2))
        return output


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in1_features': 4, 'in2_features': 4, 'num_label': 4}]
