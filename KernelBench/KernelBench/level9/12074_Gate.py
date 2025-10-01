from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
import torch.nn.functional as F


class Gate(nn.Module):

    def __init__(self, args):
        super(Gate, self).__init__()
        self.d_model = args.d_model
        self.weight_proj = nn.Linear(2 * self.d_model, 1)
        self.tanh = nn.Tanh()

    def forward(self, featureA, featureB):
        feature = torch.cat([featureA, featureB], dim=-1)
        att = self.tanh(self.weight_proj(feature))
        gate_score = F.sigmoid(att)
        gate_score = gate_score.repeat(1, 1, self.d_model)
        return gate_score * featureA + (1 - gate_score) * featureB


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'args': _mock_config(d_model=4)}]
