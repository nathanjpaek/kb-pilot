import torch
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class ActionAttentionV2(nn.Module):

    def __init__(self, model_dim, n_actions):
        super(ActionAttentionV2, self).__init__()
        self.model_dim = model_dim
        self.n_actions = n_actions
        self.fcq = nn.Linear(model_dim, model_dim)
        self.fck = nn.Linear(model_dim, model_dim)
        self.fca = nn.Linear(model_dim, n_actions)

    def forward(self, queries, keys):
        model_dim = self.model_dim
        n_actions = self.n_actions
        batch_size = queries.size(0)
        a = self.fca(queries)
        q = self.fcq(queries).view(batch_size, 1, model_dim)
        k = self.fck(keys).view(batch_size, -1, model_dim)
        v = th.bmm(q, k.transpose(1, 2)) / np.sqrt(model_dim)
        v = F.softmax(v, dim=-1)
        v = th.bmm(v.transpose(1, 2), a.view(batch_size, 1, n_actions))
        return v


def get_inputs():
    return [torch.rand([4, 1, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'model_dim': 4, 'n_actions': 4}]
