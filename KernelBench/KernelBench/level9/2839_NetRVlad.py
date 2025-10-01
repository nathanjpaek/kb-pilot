import torch
import torch.nn as nn


def _moveaxis(tensor: 'torch.Tensor', source: 'int', destination: 'int'
    ) ->torch.Tensor:
    dim = tensor.dim()
    perm = list(range(dim))
    if destination < 0:
        destination += dim
    perm.pop(source)
    perm.insert(destination, source)
    return tensor.permute(*perm)


class NetRVlad(nn.Module):

    def __init__(self, input_size, nb_cluster, dim=1, flatten=True):
        super().__init__()
        self.nb_cluster = nb_cluster
        self.assignement_fc = nn.Linear(input_size, nb_cluster)
        self.dim = dim
        self.flatten = flatten

    def forward(self, x):
        feat = x.shape[-1]
        x = _moveaxis(x, self.dim, -2)
        a = torch.softmax(self.assignement_fc(x), dim=-1)
        a_x = torch.einsum('...ij,...ik->...jk', a, x)
        x = a_x / a.sum(-2).unsqueeze(-1)
        if self.flatten:
            x = x.view(*x.shape[:-2], self.nb_cluster * feat)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'nb_cluster': 4}]
