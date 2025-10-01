import torch
import torch.nn as nn


class Batch33MatVec3Mul(nn.Module):

    def __init(self):
        super().__init__()

    def forward(self, mat, vec):
        vec = vec.unsqueeze(2)
        result = torch.matmul(mat, vec)
        return result.squeeze(2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
