from _paritybench_helpers import _mock_config
from torch.nn import Module
import torch
import torch.nn as nn
import torch.nn


class ConcatClassifierHead(Module):

    def __init__(self, config: 'dict'):
        super(ConcatClassifierHead, self).__init__()
        self.linear_layer_1 = nn.Linear(config['max_objects_per_scene'] *
            config['hidden_dim'], config['hidden_dim'])
        self.linear_layer_2 = nn.Linear(config['hidden_dim'], config[
            'num_output_classes'])

    def forward(self, input_set):
        flat_set = input_set.view(input_set.size(0), -1)
        flat_set = nn.ReLU()(self.linear_layer_1(flat_set))
        return self.linear_layer_2(flat_set)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(max_objects_per_scene=4, hidden_dim
        =4, num_output_classes=4)}]
