from torch.nn import Module
import torch
from torch import Tensor
from abc import abstractmethod
from typing import Tuple
import torch.nn as nn
from typing import Dict
import torch.utils.data.distributed
from torch.nn import CrossEntropyLoss
from torch.backends import cudnn as cudnn
from torch.nn import BCELoss


class Module(nn.Module):
    """
    Generic building block, which assumes to have trainable parameters within it.

    This extension allows to group the layers and have an easy access to them via group names.

    """

    def __init__(self, input_shape=None, output_shape=None):
        super(Module, self).__init__()
        self.__param_groups = dict()
        self.optimize_cb = None
        self.__input_shape = input_shape
        self.__output_shape = output_shape

    def validate_input(self, x):
        if self.__input_shape is not None:
            if len(x.shape) != len(self.__input_shape):
                raise ValueError('Expect {}-dim input, but got {}'.format(
                    len(x.shape), len(self.__input_shape)))
            for i, d in enumerate(self.__input_shape):
                if d is not None and d != x.shape[i]:
                    raise ValueError(
                        f'Expect dim {i} to be {d}, but got {x.shape[i]}')

    def validate_output(self, y):
        if self.__output_shape is not None:
            if len(y.shape) != len(self.__output_shape):
                raise ValueError('Expect {}-dim input, but got {}'.format(
                    len(y.shape), len(self.__output_shape)))
            for i, d in enumerate(self.__output_shape):
                if d is not None and d != y.shape[i]:
                    raise ValueError(
                        f'Expect dim {i} to be {d}, but got {y.shape[i]}')

    def group_parameters(self, group: 'str or Tuple[str] or None'=None,
        name: 'str or None'=None) ->Dict[str, torch.nn.Parameter or str]:
        """
        Returns an iterator through the parameters of the module from one or many groups.

        Also allows to retrieve a particular module from a group using its name.

        Parameters
        ----------
        group: str or Tuple[str] or None
            Parameter group names.
        name: str or Tuple[str] or None
            Name of the module from the group to be returned. Should be set to None
            if all the parameters from the group are needed. Alternatively, multiple modules
            from the group can be returned if it is a Tuple[str].

        Yields
        -------
        Parameters: Dict[str, torch.nn.Parameter or str]
            Dictionary of parameters. Allows to get all the parameters of submodules from multiple groups,
            or particular submodules' parameters from the given group. The returned dict has always three keys:
            params (used by optimizer), name (module name) and group name (name of the parameter groups). If name is not
            specified, it will be None.

        """
        if group is None:
            yield {'params': super(Module, self).parameters(), 'name': None,
                'group_name': None}
        elif name is None:
            if isinstance(group, str):
                group = group,
            for group_name in group:
                yield {'params': self.__param_groups[group_name], 'name':
                    None, 'group_name': group_name}
        else:
            if not isinstance(group, str):
                raise ValueError
            if isinstance(name, str):
                name = name,
            for module_name in name:
                yield {'params': self.__param_groups[group][module_name],
                    'name': module_name, 'group_name': group}

    def add_to(self, layer: 'torch.nn.Module', name: 'str', group_names:
        'str or Tuple[str]'):
        """
        Adds a layer with trainable parameters to one or several groups.

        Parameters
        ----------
        layer : torch.nn.Module
            The layer to be added to the group(s)
        name : str
            Name of the layer
        group_names: str Tuple[str]
            Group names.

        """
        if name is None or group_names is None:
            raise ValueError
        for group_name in group_names:
            if group_name not in self.__param_groups:
                self.__param_groups[group_name] = {}
            self.__param_groups[group_name][name] = layer.parameters()

    @abstractmethod
    def forward(self, *x):
        raise NotImplementedError

    @abstractmethod
    def get_features(self):
        raise NotImplementedError

    @abstractmethod
    def get_features_by_name(self, name: 'str'):
        raise NotImplementedError

    def initialize(self):

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.apply(init_weights)


class SSDicriminatorLoss(Module):

    def __init__(self, alpha=0.5):
        super().__init__()
        self.__loss_valid = BCELoss()
        self.__loss_cls = CrossEntropyLoss()
        self.__alpha = alpha

    def forward(self, pred: 'Tensor', target: 'Tensor'):
        pred_valid = pred[:, -1]
        pred_cls = pred[:, 0:-1]
        if len(target.shape) > 1 and target.shape[1] > 1:
            target_valid = target[:, -1]
            target_cls = target[:, 0:-1]
            loss_valid = self.__loss_valid(pred_valid, target_valid)
            decoded_target_cls = target_cls.argmax(dim=-1)
            loss_cls = self.__loss_cls(pred_cls, decoded_target_cls)
            _loss = self.__alpha * loss_valid + (1 - self.__alpha) * loss_cls
        else:
            target_valid = target
            _loss = self.__loss_valid(pred_valid, target_valid)
        return _loss


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
