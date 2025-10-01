import torch
import torch as T
import torch.nn as nn


class LinearEnsemble(nn.Module):
    __constants__ = ['in_features', 'out_features']
    ensemble_size: 'int'
    in_features: 'int'
    out_features: 'int'
    weight: 'T.Tensor'

    def __init__(self, ensemble_size: 'int', in_features: 'int',
        out_features: 'int', weight_decay: 'float'=0.0, bias: 'bool'=True
        ) ->None:
        super(LinearEnsemble, self).__init__()
        self.ensemble_size = ensemble_size
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(T.Tensor(ensemble_size, in_features,
            out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(T.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        pass

    def forward(self, x: 'T.Tensor') ->T.Tensor:
        return T.add(T.bmm(x, self.weight), self.bias[:, None, :])

    def extra_repr(self) ->str:
        return 'in_features={}, out_features={}, bias={}'.format(self.
            in_features, self.out_features, self.bias is not None)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'ensemble_size': 4, 'in_features': 4, 'out_features': 4}]
