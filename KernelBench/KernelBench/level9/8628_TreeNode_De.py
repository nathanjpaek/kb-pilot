from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):

    def __init__(self, options, weights=None):
        super(Linear, self).__init__()
        self.n_in = options['n_in']
        self.n_out = options['n_out']
        self.layer = nn.Linear(self.n_in, self.n_out)
        if weights is not None:
            self.layer.weight = nn.Parameter(self.load(weights))
        else:
            nn.init.xavier_normal_(self.layer.weight)
        self.fixed = options['fixed']
        if self.fixed:
            self.layer.weight.requires_grad = False

    def forward(self, input):
        return self.layer(input)


class TreeNode_De(nn.Module):

    def __init__(self, options):
        super(TreeNode_De, self).__init__()
        self.layer = Linear(options)

    def forward(self, data):
        return F.relu(self.layer(data))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'options': _mock_config(n_in=4, n_out=4, fixed=4)}]
