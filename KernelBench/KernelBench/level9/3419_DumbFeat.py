from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
import torch.optim
import torch.nn.parallel


class DumbFeat(nn.Module):

    def __init__(self, opt):
        super(DumbFeat, self).__init__()
        dropout = opt['dropout'] if 'dropout' in opt else 0.0
        self.dropout = torch.nn.Dropout(p=dropout, inplace=False
            ) if dropout > 0.0 else None

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        assert x.dim() == 2
        if self.dropout is not None:
            x = self.dropout(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'opt': _mock_config(dropout=0.5)}]
