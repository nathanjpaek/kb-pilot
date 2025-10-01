import torch
import torch.nn as nn
import torch.nn.functional as F


def gram_matrix(input):
    """ gram matrix for feature assignments """
    a, b, c, d = input.size()
    allG = []
    for i in range(a):
        features = input[i].view(b, c * d)
        gram = torch.mm(features, features.t())
        gram = gram.div(c * d)
        allG.append(gram)
    return torch.stack(allG)


class GetStyleLoss(nn.Module):
    """ evaluate the style loss with gram matrix """

    def forward(self, input, target):
        """ forward pass """
        gram = gram_matrix(target)
        return F.mse_loss(gram, input)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
