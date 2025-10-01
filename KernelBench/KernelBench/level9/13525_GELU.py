import torch
import torch.nn.functional as F
import torch.utils.model_zoo
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class GELU(torch.nn.Module):

    def forward(self, x):
        return F.gelu(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
