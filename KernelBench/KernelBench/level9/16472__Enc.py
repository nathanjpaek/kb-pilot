import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class _NestedEnc(torch.nn.Module):

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class _Enc(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.e1 = _NestedEnc(torch.nn.Linear(4, 2))
        self.e2 = _NestedEnc(self.e1.f)

    def forward(self, x):
        return self.e1(x) + self.e2(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
