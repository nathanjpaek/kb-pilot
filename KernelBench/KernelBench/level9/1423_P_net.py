import torch
import torch.nn as nn
import torch.nn.functional as F


class P_net(nn.Module):

    def __init__(self, X_dim, N, z_dim):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, int(N / 2))
        self.lin2 = nn.Linear(int(N / 2), N)
        self.lin4 = nn.Linear(N, X_dim)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = self.lin4(x)
        return F.sigmoid(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'X_dim': 4, 'N': 4, 'z_dim': 4}]
