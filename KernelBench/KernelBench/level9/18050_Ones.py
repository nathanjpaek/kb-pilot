import torch
import torch.optim


class Ones(torch.nn.Module):

    def __init__(self):
        super(Ones, self).__init__()

    def forward(self, tensor: 'torch.Tensor') ->torch.Tensor:
        return torch.ones(1, *tensor.shape[1:], dtype=tensor.dtype, device=
            tensor.device).expand_as(tensor
            ) if tensor.shape else torch.scalar_tensor(1, dtype=tensor.
            dtype, device=tensor.device)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
