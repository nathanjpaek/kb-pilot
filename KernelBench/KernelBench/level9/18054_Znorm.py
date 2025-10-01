import torch
import typing
import torch.optim


def dims(tensor: 'torch.Tensor', start_index: 'int'=1) ->torch.Tensor:
    return torch.Tensor([tensor.size()[start_index:]]).squeeze()


class Znorm(torch.nn.Module):

    def __init__(self, dims: 'typing.Sequence[int]'):
        super(Znorm, self).__init__()
        self.dims = dims

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        std, mean = torch.std_mean(x, self.dims, keepdim=True)
        return (x - mean) / std


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dims': 4}]
