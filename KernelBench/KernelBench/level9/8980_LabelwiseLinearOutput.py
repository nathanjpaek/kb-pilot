import torch
import torch.nn as nn


class LabelwiseLinearOutput(nn.Module):
    """Applies a linear transformation to the incoming data for each label

    Args:
        input_size (int): The number of expected features in the input.
        num_classes (int): Total number of classes.
    """

    def __init__(self, input_size, num_classes):
        super(LabelwiseLinearOutput, self).__init__()
        self.output = nn.Linear(input_size, num_classes)

    def forward(self, input):
        return (self.output.weight * input).sum(dim=-1) + self.output.bias


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'num_classes': 4}]
