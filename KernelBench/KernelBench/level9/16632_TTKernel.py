import torch
from torch import nn
import torch.nn.functional as F


class TTKernel(nn.Module):

    def __init__(self, r_i, m, r_j):
        super(TTKernel, self).__init__()
        self.fc1 = nn.Bilinear(r_i, m, r_j, bias=False)

    def forward(self, input_tensor_1, input_tensor_2):
        tensor_transformed = self.fc1(input_tensor_1, input_tensor_2)
        return F.relu(tensor_transformed)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'r_i': 4, 'm': 4, 'r_j': 4}]
