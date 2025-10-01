from torch.nn import Module
import torch
import torch.utils.data
import torch.nn.functional
import torch.autograd


class KLDivLoss(Module):
    """
    ## KL-Divergence loss

    This calculates the KL divergence between a given normal distribution and $\\mathcal{N}(0, 1)$
    """

    def forward(self, sigma_hat: 'torch.Tensor', mu: 'torch.Tensor'):
        return -0.5 * torch.mean(1 + sigma_hat - mu ** 2 - torch.exp(sigma_hat)
            )


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
