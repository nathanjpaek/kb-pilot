import torch
import torch.optim


def flatten_spatial_dims(tensor: 'torch.Tensor', spatial_start_index: 'int'=2
    ) ->torch.Tensor:
    dims = [*tensor.shape[:spatial_start_index]] + [-1]
    return tensor.view(*dims)


def dims(tensor: 'torch.Tensor', start_index: 'int'=1) ->torch.Tensor:
    return torch.Tensor([tensor.size()[start_index:]]).squeeze()


class SpatialSoftmax(torch.nn.Module):

    def __init__(self, temperature: 'float'=1.0, alpha: 'float'=1.0,
        normalize: 'bool'=False):
        super(SpatialSoftmax, self).__init__()
        self.temp = temperature
        self.alpha = alpha
        self.normalize = normalize

    def forward(self, tensor: 'torch.Tensor') ->torch.Tensor:
        reduced = flatten_spatial_dims(tensor)
        if self.temp != 1.0:
            reduced = reduced * self.temp
        if self.alpha != 1.0:
            reduced = reduced ** self.alpha
        if self.normalize:
            reduced = reduced / reduced.flatten(2).sum(-1)
        softmaxed = torch.nn.functional.softmax(reduced, dim=-1)
        return softmaxed.view_as(tensor)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
