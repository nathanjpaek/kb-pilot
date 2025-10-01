import torch
import torch as th
from torch import nn


class LearnableClsToken(nn.Module):
    """
    Layer that adds learnable CLS tokens to sequence input.
    """

    def __init__(self, d_model: 'int'):
        super().__init__()
        cls_token = th.zeros(d_model)
        self.cls_param = nn.Parameter(cls_token, requires_grad=True)
        self.fixed_ones = nn.Parameter(th.ones(1), requires_grad=False)

    def forward(self, features, mask, lengths):
        """
        CLS Token forward.
        """
        batch, _seq_len, _d_model = features.shape
        features = th.cat([self.cls_param.unsqueeze(0).unsqueeze(0).repeat(
            batch, 1, 1), features], dim=1)
        assert th.all(features[0, 0, :] == self.cls_param)
        zeros = (self.fixed_ones.unsqueeze(0).repeat(batch, 1) * 0).bool()
        mask = th.cat([zeros, mask], dim=1)
        lengths = lengths + 1
        return features, mask, lengths


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'d_model': 4}]
