import torch
from torch import nn


class Transformation(torch.nn.Module):

    def __init__(self, input_size):
        super(Transformation, self).__init__()
        self.input_size = input_size
        self.linear1 = torch.nn.Linear(self.input_size, self.input_size)
        self.linear2 = torch.nn.Linear(self.input_size, self.input_size)
        self.linear3 = torch.nn.Linear(self.input_size, self.input_size)
        self.linear4 = torch.nn.Linear(self.input_size, self.input_size)
        self.leaky_relu = nn.LeakyReLU(0.3)

    def forward(self, x):
        """
        Transforms input x with a mask M(x) followed by multiplication with x.
        """
        h = self.linear1(x.float())
        h = self.leaky_relu(h)
        h = self.linear2(h)
        h = self.leaky_relu(h)
        h = self.linear3(h)
        h = self.leaky_relu(h)
        h = self.linear4(h)
        m = torch.sigmoid(h)
        t_x = m * x.float()
        return t_x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
