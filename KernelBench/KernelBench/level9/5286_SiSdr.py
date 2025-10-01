import torch
from torch import Tensor
from torch import nn


class SiSdr(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input: 'Tensor', target: 'Tensor'):
        eps = torch.finfo(input.dtype).eps
        Rss: 'Tensor' = torch.einsum('bi,bi->b', target, target).unsqueeze(-1)
        a: 'Tensor' = torch.einsum('bi,bi->b', target, input).add(eps
            ).unsqueeze(-1) / Rss.add(eps)
        e_true = a * target
        e_res = input - e_true
        Sss = e_true.square()
        Snn = e_res.square()
        Sss = Sss.sum(-1)
        Snn = Snn.sum(-1)
        return 10 * torch.log10(Sss.add(eps) / Snn.add(eps))


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
