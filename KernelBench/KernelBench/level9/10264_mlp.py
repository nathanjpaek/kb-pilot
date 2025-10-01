import torch
import torch.nn as nn


class mlp(nn.Module):

    def __init__(self, seq_len):
        super(mlp, self).__init__()
        self.lin1 = nn.Linear(seq_len, 2048)
        self.lin2 = nn.Linear(2048, 2048)
        self.lin3 = nn.Linear(2048, seq_len)
        self.relu = nn.ReLU()

    def forward(self, input_):
        input_ = input_.reshape(input_.size(0), -1)
        out = self.lin1(input_)
        out = self.lin2(self.relu(out))
        out = self.lin3(self.relu(out))
        return out.view(input_.size(0), -1)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'seq_len': 4}]
