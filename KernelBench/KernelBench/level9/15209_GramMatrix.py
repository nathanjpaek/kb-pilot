import torch
import torch.nn as nn


def get_outnorm(x: 'torch.Tensor', out_norm: 'str'='') ->torch.Tensor:
    """ Common function to get a loss normalization value. Can
        normalize by either the batch size ('b'), the number of
        channels ('c'), the image size ('i') or combinations
        ('bi', 'bci', etc)
    """
    img_shape = x.shape
    if not out_norm:
        return 1
    norm = 1
    if 'b' in out_norm:
        norm /= img_shape[0]
    if 'c' in out_norm:
        norm /= img_shape[-3]
    if 'i' in out_norm:
        norm /= img_shape[-1] * img_shape[-2]
    return norm


class GramMatrix(nn.Module):

    def __init__(self, out_norm: 'str'='ci'):
        """ Gram Matrix calculation.
        Args:
            out_norm: normalizes the Gram matrix. It depends on the
                implementation, according to:
                - the number of elements in each feature map channel ('i')
                - Johnson et al. (2016): the total number of elements ('ci')
                - Gatys et al. (2015): not normalizing ('')
        """
        super().__init__()
        self.out_norm = out_norm

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Calculate Gram matrix (x * x.T).
        Args:
            x: Tensor with shape of (b, c, h, w).
        Returns:
            Gram matrix of the tensor.
        """
        norm = get_outnorm(x, self.out_norm)
        mat = x.flatten(-2)
        gram = mat @ mat.transpose(-2, -1)
        return gram * norm


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
