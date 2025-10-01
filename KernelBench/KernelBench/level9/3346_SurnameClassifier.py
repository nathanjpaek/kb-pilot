from torch.nn import Module
import torch
from torch.nn import Linear
from torch.nn.functional import softmax
from torch.nn.functional import relu
from torch.nn.functional import dropout


class SurnameClassifier(Module):

    def __init__(self, input_dim: 'int', hidden_dim: 'int', output_dim: 'int'
        ) ->None:
        super().__init__()
        self.fc1 = Linear(input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, output_dim)

    def forward(self, x_in, apply_activator: 'bool'=False):
        intermediate_vector = relu(self.fc1(x_in))
        prediction_vector = self.fc2(dropout(intermediate_vector, p=0.5))
        if apply_activator:
            prediction_vector = softmax(prediction_vector, dim=1)
        return prediction_vector


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4}]
