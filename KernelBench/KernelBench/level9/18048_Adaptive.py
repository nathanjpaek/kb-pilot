import torch
import torch.optim


def dims(tensor: 'torch.Tensor', start_index: 'int'=1) ->torch.Tensor:
    return torch.Tensor([tensor.size()[start_index:]]).squeeze()


class Adaptive(torch.nn.Module):

    def __init__(self, scale_factor: 'float'=2.0, mode: 'str'='max', dims:
        'int'=2):
        super(Adaptive, self).__init__()
        self.pool_func = getattr(torch.nn.functional,
            f'adaptive_{mode}_pool{dims}d')
        self.scale_factor = scale_factor
        self.dims = dims

    def even_size(self, size: 'int', scale_factor: 'float') ->int:
        downscaled = int(size // scale_factor)
        return downscaled + int(downscaled % 2)

    def forward(self, tensor: 'torch.Tensor') ->torch.Tensor:
        size = [self.even_size(s, self.scale_factor) for s in tensor.shape[
            2:2 + self.dims]]
        return self.pool_func(tensor, size)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
