import torch
import torch.nn as nn


class NormalLoss(nn.Module):

    def __init__(self):
        super(NormalLoss, self).__init__()

    def forward(self, grad_fake, grad_real):
        prod = (grad_fake[:, :, None, :] @ grad_real[:, :, :, None]).squeeze(-1
            ).squeeze(-1)
        fake_norm = torch.sqrt(torch.sum(grad_fake ** 2, dim=-1))
        real_norm = torch.sqrt(torch.sum(grad_real ** 2, dim=-1))
        return 1 - torch.mean(prod / (fake_norm * real_norm))


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
