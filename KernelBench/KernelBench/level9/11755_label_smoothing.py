import torch
import torch.nn as nn


class label_smoothing(nn.Module):

    def __init__(self, epsilon=0.1):
        """Applies label smoothing. See https://arxiv.org/abs/1512.00567.

        Args:
            epsilon: Smoothing rate.
        """
        super(label_smoothing, self).__init__()
        self.epsilon = epsilon

    def forward(self, inputs):
        K = inputs.size()[-1]
        return (1 - self.epsilon) * inputs + self.epsilon / K


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
