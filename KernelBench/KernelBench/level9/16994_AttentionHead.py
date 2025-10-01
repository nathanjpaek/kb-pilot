import math
import torch
import torch.utils.data
import torch.nn as nn


class AttentionHead(nn.Module):

    def __init__(self, d_model, d_k, d_v, device):
        super(AttentionHead, self).__init__()
        self.dk = math.sqrt(d_k)
        self.query_layer = nn.Linear(d_model, d_k)
        self.key_layer = nn.Linear(d_model, d_k)
        self.value_layer = nn.Linear(d_model, d_v)
        self

    def forward(self, input_query, input_key, input_value):
        query = self.query_layer(input_query)
        key = torch.transpose(self.key_layer(input_key), 1, 2)
        value = self.value_layer(input_value)
        score = torch.matmul(query, key)
        score = torch.nn.functional.softmax(score, dim=2)
        z = torch.matmul(score, value)
        return z


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'd_k': 4, 'd_v': 4, 'device': 0}]
