import torch
import torch.optim.lr_scheduler
import torch.nn as nn
import torch.optim
import torch.onnx.operators


def masked_softmax(vector: 'torch.Tensor', mask: 'torch.Tensor', dim: 'int'
    =-1, memory_efficient: 'bool'=False, mask_fill_value: 'float'=-1e+32
    ) ->torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(),
                mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


class Generator(torch.nn.Module):
    """
    An ``Attention`` takes two inputs: a (batched) vector and a matrix, plus an optional mask on the
    rows of the matrix.  We compute the similarity between the vector and each row in the matrix,
    and then (optionally) perform a softmax over rows using those computed similarities.
    Inputs:
    - vector: shape ``(batch_size, embedding_dim)``
    - matrix: shape ``(batch_size, num_rows, embedding_dim)``
    - matrix_mask: shape ``(batch_size, num_rows)``, specifying which rows are just padding.
    Output:
    - attention: shape ``(batch_size, num_rows)``.
    Parameters
    ----------
    normalize : ``bool``, optional (default: ``True``)
        If true, we normalize the computed similarities with a softmax, to return a probability
        distribution for your attention.  If false, this is just computing a similarity score.
    """

    def __init__(self, tensor_1_dim: 'int', tensor_2_dim: 'int'):
        super(Generator, self).__init__()
        self.project = nn.Linear(in_features=tensor_1_dim, out_features=
            tensor_2_dim)

    def forward(self, vector: 'torch.Tensor', matrix: 'torch.Tensor',
        matrix_mask: 'torch.Tensor'=None) ->torch.Tensor:
        trans_vec = self.project(vector)
        _batch, length, _dim = matrix.size()
        new_vec = torch.unsqueeze(trans_vec, dim=2).expand(-1, -1, length)
        new_vec = new_vec.transpose(1, 2)
        product = new_vec * matrix
        similarities = torch.sum(product, dim=2)
        probs = masked_softmax(similarities, matrix_mask)
        return probs


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'tensor_1_dim': 4, 'tensor_2_dim': 4}]
