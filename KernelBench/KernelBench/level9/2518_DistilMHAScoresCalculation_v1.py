import math
import torch
import torch.nn as nn
import torch.cuda
import torch.backends.cudnn
import torch.backends.mkl


class DistilMHAScoresCalculation_v1(nn.Module):

    def __init__(self, dim_per_head, softmax_dim=-1):
        super(DistilMHAScoresCalculation_v1, self).__init__()
        self.softmax = nn.Softmax(dim=softmax_dim)
        self.dim_per_head = dim_per_head

    def forward(self, mat1, mat2, mask):
        mask_shape = [mat1.shape[0], 1, 1, mat1.shape[3]]
        mat1 = mat1 / math.sqrt(self.dim_per_head)
        qk = torch.matmul(mat1, mat2.transpose(2, 3))
        mask = (mask == 0).view(mask_shape).expand_as(qk)
        qk.masked_fill_(mask, -float('inf'))
        return self.softmax(qk)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 1, 1, 4])]


def get_init_inputs():
    return [[], {'dim_per_head': 4}]
