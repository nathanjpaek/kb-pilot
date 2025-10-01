import torch
import torch.nn as nn


class CustomNet(nn.Module):
    """
    A network with a fully connected layer followed by a sigmoid layer. This is
    used for testing customized operation handles.
    """

    def __init__(self, input_dim: 'int', output_dim: 'int') ->None:
        super(CustomNet, self).__init__()
        self.conv = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.conv(x)
        x = self.sigmoid(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
