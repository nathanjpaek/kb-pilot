from _paritybench_helpers import _mock_config
from torch.nn import Module
import torch
import torch.nn as nn
import torch.nn


class PerOutputClassifierHead(Module):

    def __init__(self, config: 'dict'):
        super(PerOutputClassifierHead, self).__init__()
        self.linear_layer_1 = nn.Linear(config['hidden_dim'], config[
            'hidden_dim'] // 2)
        self.linear_layer_2 = nn.Linear(config['hidden_dim'] // 2, config[
            'num_output_classes'])

    def forward(self, input_set):
        reduced_set = torch.sum(input_set, dim=1)
        reduced_set = self.linear_layer_1(reduced_set)
        reduced_set = nn.ReLU()(reduced_set)
        reduced_set = self.linear_layer_2(reduced_set)
        return reduced_set


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_dim=4, num_output_classes=4)}]
