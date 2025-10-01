import torch
import torch.nn as nn


class learned_similarity_8(nn.Module):

    def __init__(self, in_size=1024):
        super(learned_similarity_8, self).__init__()
        self.lin = nn.Linear(1, 1)
        self.lin2 = nn.Linear(1, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, xi, xj):
        out = torch.reshape(torch.dist(xi, xj, 2), (1, 1))
        out = self.lin2(out)
        out = self.tanh(out)
        out = self.lin(out)
        out = self.sigmoid(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
