import torch
import torch.nn as nn
import torch.utils.data


class KLDLoss(nn.Module):

    def forward(self, mu1, logvar1, mu2, logvar2):
        batch_size = mu1.shape[0]
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2 / sigma1 + 1e-08) + (torch.exp(logvar1) + (
            mu1 - mu2) ** 2) / (2 * torch.exp(logvar2) + 1e-08) - 1 / 2
        return kld.sum() / batch_size


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
