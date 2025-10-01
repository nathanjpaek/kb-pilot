import torch
from typing import Any
from typing import Dict
from typing import Optional
import torch.nn as nn
import torch.nn.modules as nn
import torch.optim
from torch import nn


def is_pos_int(number):
    """
    Returns True if a number is a positive integer.
    """
    return type(number) == int and number >= 0


class ClassyHead(nn.Module):
    """
    Base class for heads that can be attached to :class:`ClassyModel`.

    A head is a regular :class:`torch.nn.Module` that can be attached to a
    pretrained model. This enables a form of transfer learning: utilizing a
    model trained for one dataset to extract features that can be used for
    other problems. A head must be attached to a :class:`models.ClassyBlock`
    within a :class:`models.ClassyModel`.
    """

    def __init__(self, unique_id: 'Optional[str]'=None, num_classes:
        'Optional[int]'=None):
        """
        Constructs a ClassyHead.

        Args:
            unique_id: A unique identifier for the head. Multiple instances of
                the same head might be attached to a model, and unique_id is used
                to refer to them.

            num_classes: Number of classes for the head.
        """
        super().__init__()
        self.unique_id = unique_id or self.__class__.__name__
        self.num_classes = num_classes

    @classmethod
    def from_config(cls, config: 'Dict[str, Any]') ->'ClassyHead':
        """Instantiates a ClassyHead from a configuration.

        Args:
            config: A configuration for the ClassyHead.

        Returns:
            A ClassyHead instance.
        """
        raise NotImplementedError

    def forward(self, x):
        """
        Performs inference on the head.

        This is a regular PyTorch method, refer to :class:`torch.nn.Module` for
        more details
        """
        raise NotImplementedError


class FullyConnectedHead(ClassyHead):
    """This head defines a 2d average pooling layer
    (:class:`torch.nn.AdaptiveAvgPool2d`) followed by a fully connected
    layer (:class:`torch.nn.Linear`).
    """

    def __init__(self, unique_id: 'str', num_classes: 'int', in_plane:
        'int', zero_init_bias: 'bool'=False):
        """Constructor for FullyConnectedHead

        Args:
            unique_id: A unique identifier for the head. Multiple instances of
                the same head might be attached to a model, and unique_id is used
                to refer to them.

            num_classes: Number of classes for the head. If None, then the fully
                connected layer is not applied.

            in_plane: Input size for the fully connected layer.
        """
        super().__init__(unique_id, num_classes)
        assert num_classes is None or is_pos_int(num_classes)
        assert is_pos_int(in_plane)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = None if num_classes is None else nn.Linear(in_plane,
            num_classes)
        if zero_init_bias:
            self.fc.bias.data.zero_()

    @classmethod
    def from_config(cls, config: 'Dict[str, Any]') ->'FullyConnectedHead':
        """Instantiates a FullyConnectedHead from a configuration.

        Args:
            config: A configuration for a FullyConnectedHead.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A FullyConnectedHead instance.
        """
        num_classes = config.get('num_classes', None)
        in_plane = config['in_plane']
        return cls(config['unique_id'], num_classes, in_plane,
            zero_init_bias=config.get('zero_init_bias', False))

    def forward(self, x):
        out = self.avgpool(x)
        out = out.flatten(start_dim=1)
        if self.fc is not None:
            out = self.fc(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'unique_id': 4, 'num_classes': 4, 'in_plane': 4}]
