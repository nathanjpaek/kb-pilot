import torch
import torch as th
import torch.nn.parallel
import torch.utils.data


class PixelwiseNorm(th.nn.Module):

    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-08):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = x.pow(2.0).mean(dim=1, keepdim=True).add(alpha).sqrt()
        y = x / y
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
