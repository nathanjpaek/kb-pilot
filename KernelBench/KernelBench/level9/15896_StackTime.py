import torch
from torchvision.models.quantization import *


class StackTime(torch.nn.Module):
    __constants__ = ['factor']

    def __init__(self, factor):
        super().__init__()
        self.factor = int(factor)

    def forward(self, x, x_lens):
        r = torch.transpose(x, 0, 1)
        s = r.shape
        zeros = torch.zeros(s[0], -s[1] % self.factor, s[2], dtype=r.dtype,
            device=r.device)
        r = torch.cat([r, zeros], 1)
        s = r.shape
        rs = [s[0], s[1] // self.factor, s[2] * self.factor]
        r = torch.reshape(r, rs)
        rt = torch.transpose(r, 0, 1)
        x_lens = torch.ceil(x_lens.float() / self.factor).int()
        return rt, x_lens


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'factor': 4}]
