import torch
from typing import Union
import torch.nn as nn
from typing import Dict
import torch.utils.data


class ATOCAttentionUnit(nn.Module):
    """
    Overview:
        the attention unit of the atoc network. We now implement it as two-layer MLP, same as the original paper

    Interface:
        __init__, forward

    .. note::

        "ATOC paper: We use two-layer MLP to implement the attention unit but it is also can be realized by RNN."

    """

    def __init__(self, thought_size: 'int', embedding_size: 'int') ->None:
        """
        Overview:
            init the attention unit according to the size of input args

        Arguments:
            - thought_size (:obj:`int`): the size of input thought
            - embedding_size (:obj:`int`): the size of hidden layers
        """
        super(ATOCAttentionUnit, self).__init__()
        self._thought_size = thought_size
        self._hidden_size = embedding_size
        self._output_size = 1
        self._act1 = nn.ReLU()
        self._fc1 = nn.Linear(self._thought_size, self._hidden_size, bias=True)
        self._fc2 = nn.Linear(self._hidden_size, self._hidden_size, bias=True)
        self._fc3 = nn.Linear(self._hidden_size, self._output_size, bias=True)
        self._act2 = nn.Sigmoid()

    def forward(self, data: 'Union[Dict, torch.Tensor]') ->torch.Tensor:
        """
        Overview:
            forward method take the thought of agents as input and output the prob of these agent\\
                being initiator

        Arguments:
            - x (:obj:`Union[Dict, torch.Tensor`): the input tensor or dict contain the thoughts tensor
            - ret (:obj:`torch.Tensor`): the output initiator prob

        """
        x = data
        if isinstance(data, Dict):
            x = data['thought']
        x = self._fc1(x)
        x = self._act1(x)
        x = self._fc2(x)
        x = self._act1(x)
        x = self._fc3(x)
        x = self._act2(x)
        return x.squeeze(-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'thought_size': 4, 'embedding_size': 4}]
