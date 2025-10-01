import torch
import torch.nn as nn


class Swish(nn.Module):

    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


class Net(torch.nn.Module):

    def __init__(self, n_feature, n_hidden):
        super(Net, self).__init__()
        self.features = nn.Sequential()
        self.features.add_module('hidden', torch.nn.Linear(n_feature, n_hidden)
            )
        self.features.add_module('active1', Swish())
        self.features.add_module('hidden2', torch.nn.Linear(n_hidden, n_hidden)
            )
        self.features.add_module('active2', Swish())
        self.features.add_module('hidden3', torch.nn.Linear(n_hidden, n_hidden)
            )
        self.features.add_module('active3', Swish())
        self.features.add_module('predict', torch.nn.Linear(n_hidden, 3))

    def forward(self, x):
        return self.features(x)

    def reset_parameters(self, verbose=False):
        for module in self.modules():
            if isinstance(module, self.__class__):
                continue
        if 'reset_parameters' in dir(module):
            if callable(module.reset_parameters):
                module.reset_parameters()
            if verbose:
                None


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_feature': 4, 'n_hidden': 4}]
