import torch
import torch.nn as nn
import torch.nn.functional as F


class AttModel(nn.Module):

    def __init__(self, din, hidden_dim, dout):
        super(AttModel, self).__init__()
        self.fcv = nn.Linear(din, hidden_dim)
        self.fck = nn.Linear(din, hidden_dim)
        self.fcq = nn.Linear(din, hidden_dim)
        self.fcout = nn.Linear(hidden_dim, dout)

    def forward(self, x, mask):
        v = F.relu(self.fcv(x))
        q = F.relu(self.fcq(x))
        k = F.relu(self.fck(x)).permute(0, 2, 1)
        att = F.softmax(torch.mul(torch.bmm(q, k), mask) - 
            9000000000000000.0 * (1 - mask), dim=2)
        out = torch.bmm(att, v)
        out = F.relu(self.fcout(out))
        return out


class DGN(nn.Module):

    def __init__(self, n_agent, num_inputs, hidden_dim, num_actions):
        super(DGN, self).__init__()
        self.encoder = nn.Linear(num_inputs, hidden_dim)
        self.att_1 = AttModel(hidden_dim, hidden_dim, hidden_dim)
        self.att_2 = AttModel(hidden_dim, hidden_dim, hidden_dim)
        self.q_net = nn.Linear(hidden_dim, num_actions)

    def forward(self, x, mask):
        h1 = F.relu(self.encoder(x))
        h2 = self.att_1(h1, mask)
        h3 = self.att_2(h2, mask)
        q = self.q_net(h3)
        return q


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'n_agent': 4, 'num_inputs': 4, 'hidden_dim': 4,
        'num_actions': 4}]
