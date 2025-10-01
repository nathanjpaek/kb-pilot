import torch
import torch.nn as nn
import torch.nn.functional as F


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G / (a * b * c * d)


class Gram_StyleLoss(nn.Module):

    def __init__(self):
        super(Gram_StyleLoss, self).__init__()

    def forward(self, input, target):
        value = torch.tensor(0.0).type_as(input[0])
        for in_m, in_n in zip(input, target):
            G = gram_matrix(in_m)
            T = gram_matrix(in_n)
            value += F.mse_loss(G, T)
        return value


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4]), torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
