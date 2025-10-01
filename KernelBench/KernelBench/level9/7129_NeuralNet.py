import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNet(nn.Module):

    def __init__(self, num_input_nodes, num_hidden_nodes, output_dimension):
        super(NeuralNet, self).__init__()
        self.input_linear = nn.Linear(num_input_nodes, num_hidden_nodes)
        self.output_linear = nn.Linear(num_hidden_nodes, output_dimension)

    def forward(self, input_vector):
        out = self.input_linear(input_vector)
        out = F.tanh(out)
        out = self.output_linear(out)
        out = F.softmax(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_input_nodes': 4, 'num_hidden_nodes': 4,
        'output_dimension': 4}]
