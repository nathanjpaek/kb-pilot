import torch
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F


def get_activation(activ):
    if callable(activ):
        return activ
    if activ is None:
        return lambda x: x
    if activ == 'tanh':
        return F.tanh
    elif activ == 'relu':
        return F.relu
    elif activ == 'gelu':
        return F.gelu
    elif activ == 'glu':
        return lambda x: F.glu(x, -1)
    else:
        raise ValueError('Unknown activation: {}'.format(activ))


class MultiHeadDenseLayer(nn.Module):
    """ Auto splitting or combining heads for the linear transformation. """

    def __init__(self, input_size, output_units, num_heads, activation=None,
        use_bias=True, is_output_transform=False):
        """ Initializes MultiHeadDenseLayer.

        Args:
            input_size: The input dimension.
            output_units: A int scalar or int list, indicating the transformed output units.
                It must be a int scalar when `is_output_transform` is True.
            num_heads: The head num.
            activation: A string or a callable function for activation.
            use_bias: A boolean, whether to add bias tensor.
            is_output_transform: A boolean, whether to use this layer for the output
                transformation in multi head attention.
        """
        super(MultiHeadDenseLayer, self).__init__()
        self._output_units = output_units
        self._num_heads = num_heads
        self._use_bias = use_bias
        self._is_output_transform = is_output_transform
        self._activation = activation
        self._activation_fn = get_activation(activation)
        self._flatten_output_units = tf.nest.flatten(self._output_units)
        if is_output_transform:
            assert not tf.nest.is_nested(self._output_units)
            self._kernel = torch.nn.Parameter(torch.nn.init.xavier_normal_(
                torch.empty(input_size, self._output_units)))
        else:
            self._kernel = torch.nn.Parameter(torch.nn.init.xavier_normal_(
                torch.empty(input_size, sum(self._flatten_output_units))),
                requires_grad=True)
        if self._use_bias:
            self._bias = torch.nn.Parameter(torch.zeros(sum(self.
                _flatten_output_units)), requires_grad=True)

    def compat_kernel_shape(self, input_shape):
        """ Compatible kernel for variable storage. """
        if self._is_output_transform:
            return [input_shape[-1] * input_shape[-2], self._output_units]
        return [input_shape[-1], sum(self._flatten_output_units)]

    @property
    def kernel_shape(self):
        """ The kernel shape. """
        if self._is_output_transform:
            return [self._num_heads, -1, self._output_units]
        return [-1, sum(self._flatten_output_units)]

    def forward(self, inputs):
        """ Implements ``call()`` for MultiHeadDenseLayer.

        Args:
            inputs: A float tensor of shape [batch_size, length, hidden_size]
                when output_projection is False, otherwise a float tensor of shape
                [batch_size, length, num_heads, num_units_per_head].

        Returns:
            The projected tensor with shape [batch_size, length, num_heads,
                num_units_per_head] per `self._output_units` when output_projection
                is False, otherwise [batch_size, length, output_units].
        """
        kernel = torch.reshape(self._kernel, self.kernel_shape)
        if self._is_output_transform:
            output = torch.einsum('abcd,cde->abe', inputs, kernel)
        else:
            output = torch.einsum('abc,cd->abd', inputs, kernel)
        if self._use_bias:
            output += self._bias
        if not self._is_output_transform:
            output = torch.split(output, self._flatten_output_units, dim=-1)
            output = tf.nest.map_structure(lambda x, num_units: torch.
                reshape(x, list(x.size())[:-1] + [self._num_heads, 
                num_units // self._num_heads]), output, self.
                _flatten_output_units, check_types=False)
        output = tf.nest.flatten(output)
        if self._activation_fn is not None:
            output = tf.nest.map_structure(self._activation_fn, output,
                check_types=False)
        return tf.nest.pack_sequence_as(self._output_units, output)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_units': 4, 'num_heads': 4}]
