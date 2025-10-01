import math
import torch
import torch.nn as nn


class SimilarityFunction(nn.Module):
    """
    A ``SimilarityFunction`` takes a pair of tensors with the same shape, and computes a similarity
    function on the vectors in the last dimension.  For example, the tensors might both have shape
    `(batch_size, sentence_length, embedding_dim)`, and we will compute some function of the two
    vectors of length `embedding_dim` for each position `(batch_size, sentence_length)`, returning a
    tensor of shape `(batch_size, sentence_length)`.
    The similarity function could be as simple as a dot product, or it could be a more complex,
    parameterized function.
    """
    default_implementation = 'dot_product'

    def forward(self, tensor_1: 'torch.Tensor', tensor_2: 'torch.Tensor'
        ) ->torch.Tensor:
        """
        Takes two tensors of the same shape, such as ``(batch_size, length_1, length_2,
        embedding_dim)``.  Computes a (possibly parameterized) similarity on the final dimension
        and returns a tensor with one less dimension, such as ``(batch_size, length_1, length_2)``.
        """
        raise NotImplementedError


class DotProductSimilarity(SimilarityFunction):
    """
    This similarity function simply computes the dot product between each pair of vectors, with an
    optional scaling to reduce the variance of the output elements.
    Parameters
    ----------
    scale_output : ``bool``, optional
        If ``True``, we will scale the output by ``math.sqrt(tensor.size(-1))``, to reduce the
        variance in the result.
    """

    def __init__(self, scale_output: 'bool'=False) ->None:
        super(DotProductSimilarity, self).__init__()
        self._scale_output = scale_output

    def forward(self, tensor_1: 'torch.Tensor', tensor_2: 'torch.Tensor'
        ) ->torch.Tensor:
        result = (tensor_1 * tensor_2).sum(dim=-1)
        if self._scale_output:
            result *= math.sqrt(tensor_1.size(-1))
        return result


class MatrixAttention(nn.Module):

    def __init__(self, similarity_function: 'SimilarityFunction'=None) ->None:
        super().__init__()
        self._similarity_function = (similarity_function or
            DotProductSimilarity())

    def forward(self, matrix_1: 'torch.Tensor', matrix_2: 'torch.Tensor'
        ) ->torch.Tensor:
        tiled_matrix_1 = matrix_1.unsqueeze(2).expand(matrix_1.size()[0],
            matrix_1.size()[1], matrix_2.size()[1], matrix_1.size()[2])
        tiled_matrix_2 = matrix_2.unsqueeze(1).expand(matrix_2.size()[0],
            matrix_1.size()[1], matrix_2.size()[1], matrix_2.size()[2])
        return self._similarity_function(tiled_matrix_1, tiled_matrix_2)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
