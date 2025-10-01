import math
import torch
import torch.nn as nn
import torch.cuda
import torch.distributed


def get_activation_fn(activation):
    """Return an activation function Module according to its name."""
    if activation == 'gelu':
        fn = GELU()
    elif activation == 'relu':
        fn = nn.ReLU()
    elif activation == 'tanh':
        fn = nn.Tanh()
    else:
        raise ValueError(
            'Please pass a valid                           activation function'
            )
    return fn


class GELU(nn.Module):
    """ Implementation of the gelu activation function
        :cite:`DBLP:journals/corr/HendrycksG16`

        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi)
                    * (x + 0.044715 * torch.pow(x, 3))))

        Examples::
        >>> m = GELU()
        >>> inputs = torch.randn(2)
        >>> outputs = m(inputs)
    """

    def forward(self, x):
        gelu = x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        return gelu


class BertPredictionTransform(nn.Module):
    """{Linear(h,h), Activation, LN} block."""

    def __init__(self, hidden_size):
        """
        Args:
            hidden_size (int): BERT model hidden layer size.
        """
        super(BertPredictionTransform, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = get_activation_fn('gelu')
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states (Tensor): BERT encoder output ``(B, S, H)``
        """
        hidden_states = self.layer_norm(self.activation(self.dense(
            hidden_states)))
        return hidden_states


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
