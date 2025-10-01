import torch
from torch import nn
import torch.utils.data


class AdjDecoder(nn.Module):
    u""" Decode an input (parent) feature into a left-child and a right-child feature """

    def __init__(self, feature_size, hidden_size):
        super(AdjDecoder, self).__init__()
        self.mlp = nn.Linear(feature_size, hidden_size)
        self.mlp_left = nn.Linear(hidden_size, feature_size)
        self.mlp_right = nn.Linear(hidden_size, feature_size)
        self.tanh = nn.Tanh()

    def forward(self, parent_feature):
        vector = self.mlp(parent_feature)
        vector = self.tanh(vector)
        left_feature = self.mlp_left(vector)
        left_feature = self.tanh(left_feature)
        right_feature = self.mlp_right(vector)
        right_feature = self.tanh(right_feature)
        return left_feature, right_feature


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'feature_size': 4, 'hidden_size': 4}]
