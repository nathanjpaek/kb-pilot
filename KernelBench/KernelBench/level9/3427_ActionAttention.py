import torch
import numpy as np
import torch as th
import torch.nn as nn


class ActionAttention(nn.Module):

    def __init__(self, model_dim, n_actions):
        super(ActionAttention, self).__init__()
        self.model_dim = model_dim
        self.n_actions = n_actions
        self.fcq = nn.Linear(model_dim, model_dim * n_actions)
        self.fck = nn.Linear(model_dim, model_dim * n_actions)

    def forward(self, queries, keys):
        model_dim = self.model_dim
        n_actions = self.n_actions
        batch_size = queries.size(0)
        q = self.fcq(queries).view(batch_size * n_actions, 1, model_dim)
        k = self.fck(keys).view(batch_size, -1, n_actions, model_dim
            ).transpose(1, 2).reshape(batch_size * n_actions, -1, model_dim)
        v = th.bmm(q, k.transpose(1, 2)) / np.sqrt(model_dim)
        v = v.view(batch_size, n_actions, -1).transpose(1, 2)
        return v


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'model_dim': 4, 'n_actions': 4}]
