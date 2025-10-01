import torch
import torch.nn as nn
import torch.nn.functional as F


class D_net_gauss(nn.Module):

    def __init__(self, N, z_dim):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin3 = nn.Linear(N, int(N / 2))
        self.lin4 = nn.Linear(int(N / 2), 10)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.5, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin3(x), p=0.5, training=self.training)
        x = F.relu(x)
        return F.log_softmax(self.lin4(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'N': 4, 'z_dim': 4}]
