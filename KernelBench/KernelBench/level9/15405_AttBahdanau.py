import torch
import torch.nn as nn
import torch.nn.functional as F


class AttBahdanau(torch.nn.Module):
    """
    AttBahdanau: Attention according to Bahdanau that can be used by the 
    Alignment module.
    """

    def __init__(self, q_dim, y_dim, att_dim=128):
        super().__init__()
        self.q_dim = q_dim
        self.y_dim = y_dim
        self.att_dim = att_dim
        self.Wq = nn.Linear(self.q_dim, self.att_dim)
        self.Wy = nn.Linear(self.y_dim, self.att_dim)
        self.v = nn.Linear(self.att_dim, 1)

    def forward(self, query, y):
        att = torch.tanh(self.Wq(query).unsqueeze(1) + self.Wy(y).unsqueeze(2))
        att = self.v(att).squeeze(3).transpose(2, 1)
        sim = att.max(2)[0].unsqueeze(1)
        att = F.softmax(att, dim=2)
        return att, sim


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'q_dim': 4, 'y_dim': 4}]
