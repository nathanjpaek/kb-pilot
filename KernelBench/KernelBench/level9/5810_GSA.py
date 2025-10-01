import torch
from torch import nn


class GSAHelper(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.fc_k = nn.Linear(self.d, self.d)
        self.fc_q = nn.Linear(self.d, self.d)
        self.fc_kq = nn.Linear(self.d, self.d)

    def forward(self, k, q):
        m = k.shape[0]
        k_1 = self.fc_k(k)
        q_1 = self.fc_q(q)
        kq = nn.Sigmoid()(self.fc_kq(torch.mul(k_1, q_1)))
        k_2 = torch.mul(k, kq)
        q_2 = torch.mul(q, kq)
        mul = torch.mm(k_2, torch.t(q_2)) / self.d ** (1.0 / 2)
        a = nn.Softmax()(torch.flatten(mul)).view(m, m)
        return a


class GSA(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.fc_v = nn.Linear(self.d, self.d)
        self.fc_k = nn.Linear(self.d, self.d)
        self.fc_q = nn.Linear(self.d, self.d)
        self.gsa_helper = GSAHelper(self.d)

    def forward(self, x):
        x.shape[0]
        v = self.fc_v(x)
        k = self.fc_k(x)
        q = self.fc_q(x)
        a = self.gsa_helper(k, q)
        f = torch.mm(a, v)
        return f


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d': 4}]
