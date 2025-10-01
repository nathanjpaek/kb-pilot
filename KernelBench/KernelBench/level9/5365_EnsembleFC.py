import torch
import torch.nn as nn
import torch.utils.data


class EnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: 'int'
    out_features: 'int'
    ensemble_size: 'int'
    weight: 'torch.Tensor'

    def __init__(self, in_features: 'int', out_features: 'int',
        ensemble_size: 'int', weight_decay: 'float'=0.0) ->None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.zeros(ensemble_size, in_features,
            out_features))
        self.weight_decay = weight_decay
        self.bias = nn.Parameter(torch.zeros(ensemble_size, 1, out_features))

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        assert input.shape[0] == self.ensemble_size and len(input.shape) == 3
        return torch.bmm(input, self.weight) + self.bias

    def extra_repr(self) ->str:
        return (
            'in_features={}, out_features={}, ensemble_size={}, weight_decay={}'
            .format(self.in_features, self.out_features, self.ensemble_size,
            self.weight_decay))


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4, 'ensemble_size': 4}]
