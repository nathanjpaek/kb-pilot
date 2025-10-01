import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class PredictionHead(nn.Module):
    """
    Simple classification prediction-head block to plug ontop of the 4D
    output of a CNN.
    Args:
        num_classes: the number of different classes that can be predicted.
        input_shape: the shape that input to this head will have. Expected
                      to be (batch_size, channels, height, width)
    """

    def __init__(self, num_classes, input_shape):
        super(PredictionHead, self).__init__()
        self.avgpool = nn.AvgPool2d(input_shape[2])
        self.linear = nn.Linear(input_shape[1], num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return F.log_softmax(x, 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_classes': 4, 'input_shape': [4, 4, 4]}]
