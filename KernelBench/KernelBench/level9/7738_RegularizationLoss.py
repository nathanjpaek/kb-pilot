import torch
import torch.nn as nn


class RegularizationLoss(nn.Module):

    def __init__(self, lambda_p: 'float', max_layers: 'int'):
        super().__init__()
        p_g = torch.zeros((max_layers,))
        not_halted = 1.0
        for k in range(max_layers):
            p_g[k] = lambda_p * not_halted
            not_halted = not_halted * (1 - lambda_p)
        self.p_g = nn.Parameter(p_g, requires_grad=False)
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, probas):
        probas = probas.transpose(0, 1)
        p_g = self.p_g[None, :probas.shape[1]].expand_as(probas)
        return self.kl_div(probas.log(), p_g)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'lambda_p': 4, 'max_layers': 1}]
