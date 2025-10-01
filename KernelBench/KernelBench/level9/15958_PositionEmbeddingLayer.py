import torch
import torch.nn as nn
import torch.utils.data
from typing import Dict
from typing import Tuple
from abc import ABC
from abc import abstractmethod


class BaseLayer(nn.Module, ABC):
    """
    Base Layer for the torecsys module
    """

    def __init__(self, **kwargs):
        """
        Initializer for BaseLayer

        Args:
            **kwargs: kwargs
        """
        super(BaseLayer, self).__init__()

    @property
    @abstractmethod
    def inputs_size(self) ->Dict[str, Tuple[str, ...]]:
        """
        Get inputs size of the layer

        Returns:
            Dict[str, Tuple[str, ...]]: dictionary of inputs_size
        """
        raise NotImplementedError('not implemented')

    @property
    @abstractmethod
    def outputs_size(self) ->Dict[str, Tuple[str, ...]]:
        """
        Get outputs size of the layer

        Returns:
            Dict[str, Tuple[str, ...]]: dictionary of outputs_size
        """
        raise NotImplementedError('not implemented')


class PositionEmbeddingLayer(BaseLayer):
    """
    Layer class of Position Embedding

    Position Embedding was used in Personalized Re-ranking Model :title:`Changhua Pei et al, 2019`[1], which is to
    add a trainable tensors per position to the session-based embedding features tensor.

    :Reference:

    `Changhua Pei et al, 2019. Personalized Re-ranking for Recommendation <https://arxiv.org/abs/1904.06813>`_.

    """

    @property
    def inputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'inputs': ('B', 'L', 'E')}

    @property
    def outputs_size(self) ->Dict[str, Tuple[str, ...]]:
        return {'outputs': ('B', 'L', 'E')}

    def __init__(self, max_num_position: 'int'):
        """
        Initialize PositionEmbedding
        
        Args:
            max_num_position (int): maximum number of position in a sequence
        """
        super().__init__()
        self.bias = nn.Parameter(torch.Tensor(1, max_num_position, 1))
        nn.init.normal_(self.bias)

    def forward(self, session_embed_inputs: 'torch.Tensor') ->torch.Tensor:
        """
        Forward calculation of PositionEmbedding
        
        Args:
            session_embed_inputs (T), shape = (B, L, E), data_type = torch.float: embedded feature tensors of session
        
        Returns:
            T, shape = (B, L, E), data_type = torch.float: output of PositionEmbedding
        """
        return session_embed_inputs + self.bias


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'max_num_position': 4}]
