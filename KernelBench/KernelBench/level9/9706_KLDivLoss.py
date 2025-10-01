import torch
import torch.nn as nn


class KLDivLoss(nn.Module):
    """
    ## KL-Divergence loss

    This calculates the KL divergence between a given normal distribution and $\\mathcal{N}(0, 1)$
    """

    def forward(self, sigma_hat, mu):
        return -0.5 * torch.mean(1 + sigma_hat - mu ** 2 - torch.exp(sigma_hat)
            )


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
