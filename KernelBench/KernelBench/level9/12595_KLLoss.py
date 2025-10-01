import torch
import torch.nn as nn


class KLLoss(nn.Module):

    def forward(self, mu: 'torch.Tensor', sigma: 'torch.Tensor', target_mu:
        'torch.Tensor', target_std: 'torch.Tensor'):
        std1 = target_std
        std2 = sigma
        mean1 = target_mu
        mean2 = mu
        kl = torch.log(torch.abs(std2) / torch.abs(std1)) + (std1 ** 2 + (
            mean1 - mean2) ** 2) / (2 * std2 ** 2) - 0.5
        return kl.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
