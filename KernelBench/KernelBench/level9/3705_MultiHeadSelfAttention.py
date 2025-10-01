from torch.nn import Module
import torch
from torch.nn import Dropout
from torch.nn import Linear


def masked_softmax(vector: 'torch.Tensor', mask: 'torch.Tensor', dim: 'int'=-1
    ) ->torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.

    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    In the case that the input vector is completely masked, this function returns an array
    of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of a model
    that uses categorical cross-entropy loss.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        result = torch.nn.functional.softmax(vector + (1 - mask) * -
            10000000000.0, dim=dim)
    return result


def weighted_sum(matrix: 'torch.Tensor', attention: 'torch.Tensor'
    ) ->torch.Tensor:
    """
    Takes a matrix of vectors and a set of weights over the rows in the matrix (which we call an
    "attention" vector), and returns a weighted sum of the rows in the matrix.  This is the typical
    computation performed after an attention mechanism.

    Note that while we call this a "matrix" of vectors and an attention "vector", we also handle
    higher-order tensors.  We always sum over the second-to-last dimension of the "matrix", and we
    assume that all dimensions in the "matrix" prior to the last dimension are matched in the
    "vector".  Non-matched dimensions in the "vector" must be `directly after the batch dimension`.

    For example, say I have a "matrix" with dimensions ``(batch_size, num_queries, num_words,
    embedding_dim)``.  The attention "vector" then must have at least those dimensions, and could
    have more. Both:

        - ``(batch_size, num_queries, num_words)`` (distribution over words for each query)
        - ``(batch_size, num_documents, num_queries, num_words)`` (distribution over words in a
          query for each document)

    are valid input "vectors", producing tensors of shape:
    ``(batch_size, num_queries, embedding_dim)`` and
    ``(batch_size, num_documents, num_queries, embedding_dim)`` respectively.
    """
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)


class MultiHeadSelfAttention(Module):
    """
    This class implements the key-value scaled dot product attention mechanism
    detailed in the paper `Attention is all you Need
    <https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077>`_ .

    The attention mechanism is a weighted sum of a projection V of the inputs, with respect
    to the scaled, normalised dot product of Q and K, which are also both linear projections
    of the input. This procedure is repeated for each attention head, using different parameters.

    Parameters
    ----------
    num_heads : ``int``, required.
        The number of attention heads to use.
    input_dim : ``int``, required.
        The size of the last dimension of the input tensor.
    attention_dim ``int``, required.
        The total dimension of the query and key projections which comprise the
        dot product attention function. Must be divisible by ``num_heads``.
    values_dim : ``int``, required.
        The total dimension which the input is projected to for representing the values,
        which are combined using the attention. Must be divisible by ``num_heads``.
    output_projection_dim : ``int``, optional (default = None)
        The dimensionality of the final output projection. If this is not passed
        explicitly, the projection has size `input_size`.
    attention_dropout_prob : ``float``, optional (default = 0.1).
        The dropout probability applied to the normalised attention
        distributions.
    """

    def __init__(self, input_dim: 'int', attention_dim: 'int', values_dim:
        'int', num_heads: 'int'=1, output_projection_dim: 'int'=None,
        attention_dropout_prob: 'float'=0.1) ->None:
        super(MultiHeadSelfAttention, self).__init__()
        self._num_heads = num_heads
        self._input_dim = input_dim
        self._output_dim = output_projection_dim or input_dim
        self._attention_dim = attention_dim
        self._values_dim = values_dim
        if attention_dim % num_heads != 0:
            raise ValueError(
                f'Key size ({attention_dim}) must be divisible by the number of attention heads ({num_heads}).'
                )
        if values_dim % num_heads != 0:
            raise ValueError(
                f'Value size ({values_dim}) must be divisible by the number of attention heads ({num_heads}).'
                )
        self._combined_projection = Linear(input_dim, 2 * attention_dim +
            values_dim)
        self._scale = (input_dim // num_heads) ** 0.5
        self._output_projection = Linear(values_dim, self._output_dim)
        self._attention_dropout = Dropout(attention_dropout_prob)

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    def is_bidirectional(self):
        return False

    def forward(self, inputs: 'torch.Tensor', mask: 'torch.LongTensor'=None
        ) ->torch.FloatTensor:
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps).

        Returns
        -------
        A tensor of shape (batch_size, timesteps, output_projection_dim),
        where output_projection_dim = input_dim by default.
        """
        num_heads = self._num_heads
        batch_size, timesteps, _ = inputs.size()
        if mask is None:
            mask = inputs.new_ones(batch_size, timesteps)
        combined_projection = self._combined_projection(inputs)
        queries, keys, *values = combined_projection.split(self.
            _attention_dim, -1)
        queries = queries.contiguous()
        keys = keys.contiguous()
        values = torch.cat(values, -1).contiguous()
        values_per_head = values.view(batch_size, timesteps, num_heads, int
            (self._values_dim / num_heads))
        values_per_head = values_per_head.transpose(1, 2).contiguous()
        values_per_head = values_per_head.view(batch_size * num_heads,
            timesteps, int(self._values_dim / num_heads))
        queries_per_head = queries.view(batch_size, timesteps, num_heads,
            int(self._attention_dim / num_heads))
        queries_per_head = queries_per_head.transpose(1, 2).contiguous()
        queries_per_head = queries_per_head.view(batch_size * num_heads,
            timesteps, int(self._attention_dim / num_heads))
        keys_per_head = keys.view(batch_size, timesteps, num_heads, int(
            self._attention_dim / num_heads))
        keys_per_head = keys_per_head.transpose(1, 2).contiguous()
        keys_per_head = keys_per_head.view(batch_size * num_heads,
            timesteps, int(self._attention_dim / num_heads))
        scaled_similarities = torch.bmm(queries_per_head / self._scale,
            keys_per_head.transpose(1, 2))
        attention = masked_softmax(scaled_similarities, mask.repeat(1,
            num_heads).view(batch_size * num_heads, timesteps))
        attention = self._attention_dropout(attention)
        outputs = weighted_sum(values_per_head, attention)
        outputs = outputs.view(batch_size, num_heads, timesteps, int(self.
            _values_dim / num_heads))
        outputs = outputs.transpose(1, 2).contiguous()
        outputs = outputs.view(batch_size, timesteps, self._values_dim)
        outputs = self._output_projection(outputs)
        return outputs


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'attention_dim': 4, 'values_dim': 4}]
