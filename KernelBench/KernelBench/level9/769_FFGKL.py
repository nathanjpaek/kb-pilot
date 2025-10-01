import torch
import torch.nn as nn
import torch.utils.data
import torch.utils
from matplotlib import cm as cm
from torch.nn.parallel import *
from torchvision.models import *
from torchvision.datasets import *


class FFGKL(nn.Module):
    """KL divergence between standart normal prior and fully-factorize gaussian posterior"""

    def __init__(self):
        super(FFGKL, self).__init__()

    def forward(self, mu, var):
        return -0.5 * (1 + torch.log(var) - mu.pow(2) - var).sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
