import torch
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.init


class PixelWise(torch.nn.Module):
    """ Implemented - https://arxiv.org/pdf/1710.10196.pdf """

    def __init__(self, eps=1e-08):
        super(PixelWise, self).__init__()
        self.eps = eps

    def forward(self, tensor):
        return tensor.div(tensor.pow(2).mean(1, True).add(self.eps).pow(0.5))

    def __repr__(self):
        return 'pixelwise'


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
