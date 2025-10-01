import torch
import torch.nn as nn


class MockOpposite(nn.Module):

    def __init__(self):
        """Initialize the model"""
        super(MockOpposite, self).__init__()

    def forward(self, input):
        """A single forward pass of the model. Returns input - 1.

        Args:
            input: The input to the model as a tensor of
                batch_size X input_size
        """
        output = input.clone().detach()
        output[input == 1] = 0
        output[input == 0] = 1
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
