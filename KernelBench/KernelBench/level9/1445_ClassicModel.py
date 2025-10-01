from _paritybench_helpers import _mock_config
import torch
from torch import nn


class ClassicModel(nn.Module):

    def __init__(self, config, inpt_shp):
        super(ClassicModel, self).__init__()
        self.depth = config.depth
        self.nodes = config.n_qubits
        self.config = config
        self.pre_net = nn.Linear(inpt_shp, self.nodes)
        for i in range(self.depth):
            setattr(self, f'Linear_{i}', nn.Linear(self.nodes, self.nodes))
            setattr(self, f'ReLU_{i}', nn.ReLU())
        self.output = nn.Linear(self.nodes, 2)
        None

    def forward(self, input_features):
        input_features = self.pre_net(input_features)
        for i in range(self.depth):
            input_features = getattr(self, f'Linear_{i}')(input_features)
            input_features = getattr(self, f'ReLU_{i}')(input_features)
        output = self.output(input_features)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(depth=1, n_qubits=4), 'inpt_shp': 4}]
