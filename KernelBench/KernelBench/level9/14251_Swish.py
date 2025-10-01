import torch
import torch.nn as nn


class Swish(nn.Module):
    """The swish activation function: :math:`\\mathrm{swish}(x)=x\\sigma(\\beta x)=\\frac{x}{1+e^{-\\beta x}}`.

    :param beta: The :math:`\\beta` parameter in the swish activation.
    :type beta: float
    :param trainable: Whether scalar :math:`\\beta` can be trained
    :type trainable: bool
    """

    def __init__(self, beta=1.0, trainable=False):
        super(Swish, self).__init__()
        beta = float(beta)
        self.trainable = trainable
        if trainable:
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
