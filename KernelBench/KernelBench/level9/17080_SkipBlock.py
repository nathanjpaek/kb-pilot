import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipBlock(nn.Module):

    def __init__(self, input_dim, output_dim, activation):
        """
        Skip Connection for feed-forward block based on ResNet idea:
        Refer: 
        - Youtube: https://www.youtube.com/watch?v=ZILIbUvp5lk
        - Medium: https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624 
                 Residual block function when the input and output dimensions are not same.

        """
        super().__init__()
        assert activation in ['relu', 'tanh'
            ], 'Please specify a valid activation funciton: relu or tanh'
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = F.tanh
        self.inp_transform = nn.Linear(input_dim, output_dim)
        self.out_transform = nn.Linear(output_dim, output_dim)
        self.inp_projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        y = x --> T1(x) --> ReLU(T1(x))
        z = ReLU(T2(y) + Projection(x))
        """
        y = self.activation(self.inp_transform(x))
        z = self.activation(self.out_transform(y) + self.inp_projection(x))
        return z


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4, 'activation': 'relu'}]
