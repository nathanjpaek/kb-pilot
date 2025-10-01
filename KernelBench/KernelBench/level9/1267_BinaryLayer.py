import torch
import torch.nn as nn


class BinaryLayer(nn.Module):

    def forward(self, input):
        """Forward function for binary layer

        :param input: data
        :returns: sign of data
        :rtype: Tensor

        """
        return torch.sign(input)

    def backward(self, grad_output):
        """Straight through estimator

        :param grad_output: gradient tensor
        :returns: truncated gradient tensor
        :rtype: Tensor

        """
        input = self.saved_tensors
        grad_output[input > 1] = 0
        grad_output[input < -1] = 0
        return grad_output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
