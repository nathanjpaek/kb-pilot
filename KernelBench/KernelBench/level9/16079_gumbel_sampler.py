import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data


class gumbel_sampler(nn.Module):

    def __init__(self):
        super(gumbel_sampler, self).__init__()

    def forward(self, input, noise, temperature=0.5):
        eps = 1e-20
        noise.data.add_(eps).log_().neg_()
        noise.data.add_(eps).log_().neg_()
        y = (input + noise) / temperature
        y = F.softmax(y)
        max_val, max_idx = torch.max(y, y.dim() - 1)
        y_hard = y == max_val.view(-1, 1).expand_as(y)
        y = (y_hard.float() - y).detach() + y
        return y, max_idx.view(1, -1)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
