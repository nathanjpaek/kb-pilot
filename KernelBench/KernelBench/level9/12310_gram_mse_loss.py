import torch
import torch.nn as nn


class gram_matrix(nn.Module):

    def forward(self, input):
        b, c, w, h = input.size()
        F = input.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h * w)
        return G


class gram_mse_loss(nn.Module):

    def forward(self, input, target):
        out = nn.MSELoss()(gram_matrix()(input), target)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
