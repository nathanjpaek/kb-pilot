import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """
    Reference:
        - ACL2020, Document-Level Event Role Filler Extraction using Multi-Granularity Contextualized Encoding
    """

    def __init__(self, n_in):
        super().__init__()
        self.n_in = n_in
        self.hidden2scalar1 = nn.Linear(self.n_in, 1)
        self.hidden2scalar2 = nn.Linear(self.n_in, 1)

    def forward(self, hidden1, hidden2):
        gate_alpha = torch.sigmoid(self.hidden2scalar1(hidden1) + self.
            hidden2scalar2(hidden2))
        out = gate_alpha * hidden1 + (1 - gate_alpha) * hidden2
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_in': 4}]
