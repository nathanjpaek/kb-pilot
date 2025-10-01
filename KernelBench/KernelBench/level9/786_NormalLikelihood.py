import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.utils
from matplotlib import cm as cm
from torch.nn.parallel import *
from torchvision.models import *
from torchvision.datasets import *


class NormalLikelihood(nn.Module):

    def __init__(self):
        super(NormalLikelihood, self).__init__()

    def forward(self, target, mu, var):
        loss = torch.sum(-(target - mu) ** 2 / var - np.log(2 * np.pi) -
            torch.log(var)) * 0.5
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
