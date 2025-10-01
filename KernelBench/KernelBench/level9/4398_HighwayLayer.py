import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx.operators


class HighwayLayer(nn.Module):

    def __init__(self, input_dim, transform_activation=F.relu,
        gate_activation=F.softmax, gate_bias=-2):
        super().__init__()
        self.highway_transform_activation = transform_activation
        self.highway_gate_activation = gate_activation
        self.highway_transform = nn.Linear(input_dim, input_dim)
        self.highway_gate = nn.Linear(input_dim, input_dim)
        self.highway_gate.bias.data.fill_(gate_bias)

    def forward(self, x):
        transform_output = self.highway_transform_activation(self.
            highway_transform(x))
        gate_output = self.highway_gate_activation(self.highway_gate(x))
        transformation_part = torch.mul(transform_output, gate_output)
        carry_part = torch.mul(1 - gate_output, x)
        return torch.add(transformation_part, carry_part)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
