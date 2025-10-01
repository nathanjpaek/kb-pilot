from _paritybench_helpers import _mock_config
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, opt):
        super(Attention, self).__init__()
        self.lin_u = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
        self.lin_v = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
        self.opt = opt

    def forward(self, user, item, UV_adj, VU_adj):
        user = self.lin_u(user)
        item = self.lin_v(item)
        query = user
        key = item
        value = torch.mm(query, key.transpose(0, 1))
        value = UV_adj.to_dense() * value
        value /= math.sqrt(self.opt['hidden_dim'])
        value = F.softmax(value, dim=1)
        learn_user = torch.matmul(value, key) + user
        query = item
        key = user
        value = torch.mm(query, key.transpose(0, 1))
        value = VU_adj.to_dense() * value
        value /= math.sqrt(self.opt['hidden_dim'])
        value = F.softmax(value, dim=1)
        learn_item = torch.matmul(value, key) + item
        return learn_user, learn_item


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]),
        torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'opt': _mock_config(hidden_dim=4)}]
