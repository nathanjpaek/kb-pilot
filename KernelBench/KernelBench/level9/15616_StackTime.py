import torch
import torch.nn as nn
import torch.utils.data
import torch.jit
import torch.optim
import torch.utils.collect_env
import torch.nn.parallel
import torch.utils.data.distributed


class StackTime(nn.Module):

    def __init__(self, factor):
        super().__init__()
        self.factor = int(factor)

    def forward(self, x, x_lens):
        seq = [x]
        for i in range(1, self.factor):
            tmp = torch.zeros_like(x)
            tmp[:-i, :, :] = x[i:, :, :]
            seq.append(tmp)
        x_lens = (x_lens.int() + self.factor - 1) // self.factor
        return torch.cat(seq, dim=2)[::self.factor, :, :], x_lens


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'factor': 4}]
