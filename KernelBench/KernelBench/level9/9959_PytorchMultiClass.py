import torch
import torch.nn as nn
import torch.nn.functional as F


class PytorchMultiClass(nn.Module):
    """num_features as input parameter
    attributes:
    layer_1: fully-connected layer with 32 neurons
    layer_out: fully-connected layer with 4 neurons
    softmax: softmax function
    methods:
    forward() with inputs as input parameter, perform ReLU and DropOut on the
    fully-connected layer followed by the output layer with softmax"""

    def __init__(self, num_features):
        super(PytorchMultiClass, self).__init__()
        self.layer_1 = nn.Linear(num_features, 32)
        self.layer_out = nn.Linear(32, 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.dropout(F.relu(self.layer_1(x)), training=self.training)
        x = self.layer_out(x)
        return self.softmax(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
