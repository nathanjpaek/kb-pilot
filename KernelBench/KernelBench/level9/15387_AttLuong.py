import torch
import torch.nn as nn
import torch.nn.functional as F


class AttLuong(torch.nn.Module):
    """
    AttLuong: Attention according to Luong that can be used by the 
    Alignment module.
    """

    def __init__(self, q_dim, y_dim, softmax=True):
        super().__init__()
        self.q_dim = q_dim
        self.y_dim = y_dim
        self.softmax = softmax
        self.W = nn.Linear(self.y_dim, self.q_dim)

    def forward(self, query, y):
        att = torch.bmm(query, self.W(y).transpose(2, 1))
        sim = att.max(2)[0].unsqueeze(1)
        if self.softmax:
            att = F.softmax(att, dim=2)
        return att, sim


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'q_dim': 4, 'y_dim': 4}]
