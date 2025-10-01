import torch
import numpy as np
from torch import nn
from torch import optim
from torch import relu


def weighted_mse_loss(inputs, target, sample_weight):
    if sample_weight is not None:
        return (sample_weight * (inputs - target) ** 2).mean()
    else:
        return ((inputs - target) ** 2).mean()


class RegressorNet(nn.Module):

    def __init__(self, n_dim, num_iter=100, optimizer=optim.Adam):
        super(RegressorNet, self).__init__()
        self.hid1 = nn.Linear(n_dim, 64)
        self.drop1 = nn.Dropout(0.2)
        self.hid2 = nn.Linear(64, 32)
        self.drop2 = nn.Dropout(0.2)
        self.oupt = nn.Linear(32, 1)
        self.num_iter = num_iter
        self.optimizer = optimizer(self.parameters())

    def forward(self, x):
        z = relu(self.hid1(x))
        z = self.drop1(z)
        z = relu(self.hid2(z))
        z = self.drop2(z)
        z = self.oupt(z)
        return z

    def fit(self, X, y, sample_weight=None):
        nn.init.xavier_uniform_(self.hid1.weight)
        nn.init.zeros_(self.hid1.bias)
        nn.init.xavier_uniform_(self.hid2.weight)
        nn.init.zeros_(self.hid2.bias)
        nn.init.xavier_uniform_(self.oupt.weight)
        nn.init.zeros_(self.oupt.bias)
        X = np.array(X, dtype=np.float32)
        X = torch.from_numpy(X)
        y = np.array(y, dtype=np.float32)
        y = torch.from_numpy(y)
        if sample_weight is not None:
            weights = np.array(sample_weight, dtype=np.float32)
            weights = torch.from_numpy(weights)
        else:
            weights = None
        for _ in range(self.num_iter):
            self.optimizer.zero_grad()
            output = self.forward(X)
            loss = weighted_mse_loss(inputs=output, target=y, sample_weight
                =weights)
            loss.backward()
            self.optimizer.step()
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float32)
        X = torch.from_numpy(X)
        return self.forward(X).detach().numpy()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_dim': 4}]
