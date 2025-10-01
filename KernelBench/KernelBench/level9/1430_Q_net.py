import torch
import torch.nn as nn
import torch.nn.functional as F


class Q_net(nn.Module):

    def __init__(self, X_dim, N, z_dim):
        super(Q_net, self).__init__()
        self.xdim = X_dim
        self.lin1 = nn.Linear(X_dim, N)
        self.lin3 = nn.Linear(N, int(N / 2))
        self.lin3gauss = nn.Linear(int(N / 2), z_dim)

    def forward(self, x):
        x = x.view(-1, self.xdim)
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin3(x), p=0.25, training=self.training)
        x = F.relu(x)
        xgauss = self.lin3gauss(x)
        return xgauss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'X_dim': 4, 'N': 4, 'z_dim': 4}]
