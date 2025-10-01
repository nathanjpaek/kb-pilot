import torch
from itertools import chain as chain
import torch.utils.data
import torch.nn as nn


class TransformerBasicHead(nn.Module):
    """
    BasicHead. No pool.
    """

    def __init__(self, dim_in, num_classes, dropout_rate=0.0, act_func=
        'softmax'):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(TransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(dim_in, num_classes, bias=True)
        if act_func == 'softmax':
            self.act = nn.Softmax(dim=1)
        elif act_func == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                '{} is not supported as an activationfunction.'.format(
                act_func))

    def forward(self, x):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.projection(x)
        if not self.training:
            x = self.act(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_in': 4, 'num_classes': 4}]
