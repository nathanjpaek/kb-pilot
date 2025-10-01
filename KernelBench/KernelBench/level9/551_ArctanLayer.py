import torch
import torch.nn as nn
import torch.nn
from abc import ABCMeta
from abc import abstractmethod


class Layer(nn.Module, metaclass=ABCMeta):

    def __init__(self):
        super(Layer, self).__init__()

    @abstractmethod
    def forward(self, x):
        """
        >>> do forward pass with a given input
        """
        raise NotImplementedError

    @abstractmethod
    def bound(self, l, u, W_list, m1_list, m2_list, ori_perturb_norm=None,
        ori_perturb_eps1=None, ori_perturb_eps2=None, first_layer=False):
        """
        >>> do bound calculation

        >>> l, u: the lower and upper bound of the input, of shape [batch_size, immediate_in_dim]
        >>> W_list: the transformation matrix introduced by the previous layers, of shape [batch_size, out_dim, in_dim]
        >>> m1_list, m2_list: the bias introduced by the previous layers, of shape [batch_size, in_dim]
        >>> ori_perturb_norm, ori_perturb_eps: the original perturbation, default is None
        >>> first_layer: boolean, whether or not this layer is the first layer
        """
        raise NotImplementedError


class ArctanLayer(Layer):

    def __init__(self):
        super(ArctanLayer, self).__init__()

    def forward(self, x):
        return torch.atan(x)

    def bound(self, l, u, W_list, m1_list, m2_list, ori_perturb_norm=None,
        ori_perturb_eps=None, first_layer=False):
        assert first_layer is False, 'the first layer cannot be ReLU'
        l.shape[0]
        low_bound = torch.atan(l)
        up_bound = torch.atan(u)
        return low_bound, up_bound, W_list, m1_list, m2_list


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
