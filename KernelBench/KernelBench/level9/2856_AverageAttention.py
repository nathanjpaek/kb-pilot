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


class AverageAttention(nn.Module):
    """
    Average Attention module from
    "Accelerating Neural Transformer via an Average Attention Network"
    :cite:`DBLP:journals/corr/abs-1805-00631`.

    Args:
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, model_dim, dropout=0.1, aan_useffn=False):
        self.model_dim = model_dim
        self.aan_useffn = aan_useffn
        super(AverageAttention, self).__init__()
        if aan_useffn:
            self.average_layer = PositionwiseFeedForward(model_dim,
                model_dim, dropout)
        self.gating_layer = nn.Linear(model_dim * 2, model_dim * 2)

    def cumulative_average_mask(self, batch_size, inputs_len, device):
        """
        Builds the mask to compute the cumulative average as described in
        :cite:`DBLP:journals/corr/abs-1805-00631` -- Figure 3

        Args:
            batch_size (int): batch size
            inputs_len (int): length of the inputs

        Returns:
            (FloatTensor):

            * A Tensor of shape ``(batch_size, input_len, input_len)``
        """
        triangle = torch.tril(torch.ones(inputs_len, inputs_len, dtype=
            torch.float, device=device))
        weights = torch.ones(1, inputs_len, dtype=torch.float, device=device
            ) / torch.arange(1, inputs_len + 1, dtype=torch.float, device=
            device)
        mask = triangle * weights.transpose(0, 1)
        return mask.unsqueeze(0).expand(batch_size, inputs_len, inputs_len)

    def cumulative_average(self, inputs, mask_or_step, layer_cache=None,
        step=None):
        """
        Computes the cumulative average as described in
        :cite:`DBLP:journals/corr/abs-1805-00631` -- Equations (1) (5) (6)

        Args:
            inputs (FloatTensor): sequence to average
                ``(batch_size, input_len, dimension)``
            mask_or_step: if cache is set, this is assumed
                to be the current step of the
                dynamic decoding. Otherwise, it is the mask matrix
                used to compute the cumulative average.
            layer_cache: a dictionary containing the cumulative average
                of the previous step.

        Returns:
            a tensor of the same shape and type as ``inputs``.
        """
        if layer_cache is not None:
            step = mask_or_step
            average_attention = (inputs + step * layer_cache['prev_g']) / (step
                 + 1)
            layer_cache['prev_g'] = average_attention
            return average_attention
        else:
            mask = mask_or_step
            return torch.matmul(mask, inputs)

    def forward(self, inputs, mask=None, layer_cache=None, step=None):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, input_len, model_dim)``

        Returns:
            (FloatTensor, FloatTensor):

            * gating_outputs ``(batch_size, input_len, model_dim)``
            * average_outputs average attention
                ``(batch_size, input_len, model_dim)``
        """
        batch_size = inputs.size(0)
        inputs_len = inputs.size(1)
        average_outputs = self.cumulative_average(inputs, self.
            cumulative_average_mask(batch_size, inputs_len, inputs.device) if
            layer_cache is None else step, layer_cache=layer_cache)
        if self.aan_useffn:
            average_outputs = self.average_layer(average_outputs)
        gating_outputs = self.gating_layer(torch.cat((inputs,
            average_outputs), -1))
        input_gate, forget_gate = torch.chunk(gating_outputs, 2, dim=2)
        gating_outputs = torch.sigmoid(input_gate) * inputs + torch.sigmoid(
            forget_gate) * average_outputs
        return gating_outputs, average_outputs


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'model_dim': 4}]
