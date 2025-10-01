import torch
import torch.nn as nn
import torch.cuda


class Perceptron(nn.Module):
    """ A perceptron is one linear Layer"""

    def __init__(self, input_dim: 'int'):
        """

        :param input_dim (int): size of inputs features
        """
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)

    def forward(self, x_in):
        """The forward pass of the perceptron

        :param x_in (torch.Tensor): an input data tensor
                x_in.shape should be (batch, num_features)

        :returns
            the resulting tensort. tensor.shape should be (batch,)
        """
        return torch.sigmoid(self.fc1(x_in)).squeeze()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
