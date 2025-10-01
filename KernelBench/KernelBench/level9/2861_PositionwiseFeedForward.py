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


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
        activation (str): activation function to use. ['relu', 'gelu']
        is_bert (bool): default False. When set True,
                        layer_norm will be performed on the
                        direct connection of residual block.
    """

    def __init__(self, d_model, d_ff, dropout=0.1, activation='relu',
        is_bert=False):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12 if is_bert else 1e-06
            )
        self.dropout_1 = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """Layer definition.

        Args:
            x: ``(batch_size, input_len, model_dim)``

        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """
        x_norm = self.layer_norm(x)
        inter = self.dropout_1(self.activation(self.w_1(x_norm)))
        output = self.dropout_2(self.w_2(inter))
        residual_output = output + x_norm
        return residual_output

    def update_dropout(self, dropout):
        self.dropout_1.p = dropout
        self.dropout_2.p = dropout


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'd_ff': 4}]
