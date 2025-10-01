import math
import torch
import torch.nn as nn
import torch.cuda
import torch.backends.cudnn
import torch.backends.mkl


class DistilMHAScoresCalculation_v2(nn.Module):

    def __init__(self, dim_per_head):
        super(DistilMHAScoresCalculation_v2, self).__init__()
        self.dim_per_head = dim_per_head

    def forward(self, mat1, mat2, mask):
        mask_shape = [mat1.shape[0], 1, 1, mat1.shape[3]]
        mat1 = mat1 / math.sqrt(self.dim_per_head)
        qk = torch.matmul(mat1, mat2.transpose(2, 3))
        mask = (mask == 0).view(mask_shape).expand_as(qk)
        qk = qk.masked_fill(mask, -float('inf'))
        return nn.functional.softmax(qk, dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 1, 1, 4])]


def get_init_inputs():
    return [[], {'dim_per_head': 4}]
