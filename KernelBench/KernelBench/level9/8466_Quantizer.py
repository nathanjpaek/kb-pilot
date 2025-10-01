import torch
import torch.quantization
import torch.nn as nn
import torch.utils.data


class Quantizer(nn.Module):

    def __init__(self):
        super(Quantizer, self).__init__()

    def forward(self, x, fine_tune=False):
        cur_device = x.device
        if self.training or fine_tune:
            res = x + (torch.rand(x.size(), device=cur_device) - 0.5)
        else:
            res = torch.round(x)
        return res


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
