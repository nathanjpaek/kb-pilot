import torch
import torch.nn as nn
import torch.utils.data


class FM(nn.Module):

    def __init__(self):
        super(FM, self).__init__()

    def forward(self, X):
        square_of_sum = torch.pow(torch.sum(X, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(X * X, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)
        return cross_term


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
