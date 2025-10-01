import torch
import torch.nn as nn


def chebyshev(L, X, k=3):
    if k == 1:
        return torch.sparse.mm(L, X)
    dp = [X, torch.sparse.mm(L, X)]
    for i in range(2, k):
        nxt = 2 * torch.sparse.mm(L, dp[i - 1])
        dp.append(torch.sparse.FloatTensor.add(nxt, -dp[i - 2]))
    return torch.cat(dp, dim=1)


class SCNLayer(nn.Module):

    def __init__(self, feature_size, output_size, enable_bias=True, k=1):
        super().__init__()
        self.k = k
        self.conv = nn.Linear(k * feature_size, output_size, bias=enable_bias)

    def forward(self, L, x):
        X = chebyshev(L, x, self.k)
        return self.conv(X)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'feature_size': 4, 'output_size': 4}]
