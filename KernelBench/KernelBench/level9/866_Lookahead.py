import torch
import torch.utils.data.distributed
from torch import nn
import torch.nn.functional as F


class Lookahead(nn.Module):

    def __init__(self, n_features, context):
        super(Lookahead, self).__init__()
        assert context > 0
        self.context = context
        self.n_features = n_features
        self.pad = 0, self.context - 1
        self.conv = nn.Conv1d(self.n_features, self.n_features, kernel_size
            =self.context, stride=1, groups=self.n_features, padding=0,
            bias=False)

    def forward(self, x):
        x = x.transpose(0, 1).transpose(1, 2)
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'n_features=' + str(self.
            n_features) + ', context=' + str(self.context) + ')'


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'n_features': 4, 'context': 4}]
