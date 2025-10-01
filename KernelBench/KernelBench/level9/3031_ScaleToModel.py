import torch
import torch.nn as nn
import torch.cuda
from torch import linalg as linalg


class ScaleToModel(nn.Module):

    def __init__(self, model_value_range, test_value_range):
        super(ScaleToModel, self).__init__()
        self.m_min, self.m_max = model_value_range
        self.t_min, self.t_max = test_value_range

    def forward(self, img: 'torch.Tensor'):
        """ input: [test_val_min, test_val_max] """
        img = (img - self.t_min) / (self.t_max - self.t_min)
        img = img * (self.m_max - self.m_min) + self.m_min
        return img


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'model_value_range': [4, 4], 'test_value_range': [4, 4]}]
