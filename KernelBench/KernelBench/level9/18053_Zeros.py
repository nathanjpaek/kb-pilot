import torch
import torch.optim


class Zeros(torch.nn.Module):

    def __init__(self):
        super(Zeros, self).__init__()

    def forward(self, tensor: 'torch.Tensor') ->torch.Tensor:
        return torch.zeros(1, *tensor.shape[1:], dtype=tensor.dtype, device
            =tensor.device).expand_as(tensor)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
