import copy
import torch
import torch.utils.data
from torch.utils import data as data
import torch.nn as nn
from torch.nn import init as init
from torchvision.models import vgg as vgg
from torch import autograd as autograd


class minibatch_std_concat_layer(nn.Module):

    def __init__(self, averaging='all'):
        super(minibatch_std_concat_layer, self).__init__()
        self.averaging = averaging.lower()
        if 'group' in self.averaging:
            self.n = int(self.averaging[5:])
        else:
            assert self.averaging in ['all', 'flat', 'spatial', 'none', 'gpool'
                ], 'Invalid averaging mode' % self.averaging
        self.adjusted_std = lambda x, **kwargs: torch.sqrt(torch.mean((x -
            torch.mean(x, **kwargs)) ** 2, **kwargs) + 1e-08)

    def forward(self, x):
        shape = list(x.size())
        target_shape = copy.deepcopy(shape)
        vals = self.adjusted_std(x, dim=0, keepdim=True)
        if self.averaging == 'all':
            target_shape[1] = 1
            vals = torch.mean(vals, dim=1, keepdim=True)
        elif self.averaging == 'spatial':
            if len(shape) == 4:
                vals = mean(vals, axis=[2, 3], keepdim=True)
        elif self.averaging == 'none':
            target_shape = [target_shape[0]] + [s for s in target_shape[1:]]
        elif self.averaging == 'gpool':
            if len(shape) == 4:
                vals = mean(x, [0, 2, 3], keepdim=True)
        elif self.averaging == 'flat':
            target_shape[1] = 1
            vals = torch.FloatTensor([self.adjusted_std(x)])
        else:
            target_shape[1] = self.n
            vals = vals.view(self.n, self.shape[1] / self.n, self.shape[2],
                self.shape[3])
            vals = mean(vals, axis=0, keepdim=True).view(1, self.n, 1, 1)
        vals = vals.expand(*target_shape)
        return torch.cat([x, vals], 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
