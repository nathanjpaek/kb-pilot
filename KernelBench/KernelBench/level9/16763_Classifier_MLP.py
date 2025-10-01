import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier_MLP(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Classifier_MLP, self).__init__()
        self.h1 = nn.Linear(in_dim, hidden_dim)
        self.h2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.log_softmax(self.out(x), dim=0)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'hidden_dim': 4, 'out_dim': 4}]
