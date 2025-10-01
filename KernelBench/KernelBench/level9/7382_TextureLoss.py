import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a, b, c * d)
    G = torch.bmm(features, torch.transpose(features, 1, 2))
    return G.div(b * c * d)


class TextureLoss(nn.Module):

    def __init__(self):
        super(TextureLoss, self).__init__()

    def forward(self, x, x1):
        x = gram_matrix(x)
        x1 = gram_matrix(x1)
        loss = F.mse_loss(x, x1)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
