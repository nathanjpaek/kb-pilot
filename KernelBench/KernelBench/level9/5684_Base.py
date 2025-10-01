import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Base(nn.Module):
    """docstring for Base"""

    def __init__(self, view_space, feature_space, num_actions, hidden_size):
        super(Base, self).__init__()
        self.view_space = view_space
        self.feature_space = feature_space
        self.num_actions = num_actions
        self.l1 = nn.Linear(np.prod(view_space), hidden_size)
        self.l2 = nn.Linear(feature_space[0], hidden_size)
        self.l3 = nn.Linear(num_actions, 64)
        self.l4 = nn.Linear(64, 32)

    def forward(self, input_view, input_feature, input_act_prob):
        flatten_view = input_view.reshape(-1, np.prod(self.view_space))
        h_view = F.relu(self.l1(flatten_view))
        h_emb = F.relu(self.l2(input_feature))
        emb_prob = F.relu(self.l3(input_act_prob))
        dense_prob = F.relu(self.l4(emb_prob))
        concat_layer = torch.cat([h_view, h_emb, dense_prob], dim=1)
        return concat_layer


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'view_space': 4, 'feature_space': [4, 4], 'num_actions': 4,
        'hidden_size': 4}]
