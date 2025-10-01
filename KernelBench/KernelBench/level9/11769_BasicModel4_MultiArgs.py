import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicModel4_MultiArgs(nn.Module):
    """
    Slightly modified example model from the paper
    https://arxiv.org/pdf/1703.01365.pdf
    f(x1, x2) = RELU(ReLU(x1 - 1) - ReLU(x2) / x3)
    """

    def __init__(self) ->None:
        super().__init__()

    def forward(self, input1, input2, additional_input1, additional_input2=0):
        relu_out1 = F.relu(input1 - 1)
        relu_out2 = F.relu(input2)
        relu_out2 = relu_out2.div(additional_input1)
        return F.relu(relu_out1 - relu_out2)[:, additional_input2]


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
