import torch
from torch import nn
import torch.utils.data
import torch.nn.functional
from typing import Optional
import torch.autograd


class SpacialGatingUnit(nn.Module):
    """
    ## Spatial Gating Unit

    $$s(Z) = Z_1 \\odot f_{W,b}(Z_2)$$

    where $f_{W,b}(Z) = W Z + b$ is a linear transformation along the sequence dimension,
    and $\\odot$ is element-wise multiplication.
    $Z$ is split into to parts of equal size $Z_1$ and $Z_2$ along the channel dimension (embedding dimension).
    """

    def __init__(self, d_z: 'int', seq_len: 'int'):
        """
        * `d_z` is the dimensionality of $Z$
        * `seq_len` is the sequence length
        """
        super().__init__()
        self.norm = nn.LayerNorm([d_z // 2])
        self.weight = nn.Parameter(torch.zeros(seq_len, seq_len).uniform_(-
            0.01, 0.01), requires_grad=True)
        self.bias = nn.Parameter(torch.ones(seq_len), requires_grad=True)

    def forward(self, z: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None):
        """
        * `z` is the input $Z$ of shape `[seq_len, batch_size, d_z]`
        * `mask` is is a boolean mask of shape `[seq_len, seq_len, 1]` that controls the visibility of tokens
         among each other. The last dimension of size `1` is the batch, which we have in other transformer
         implementations and was left for compatibility.
        """
        seq_len = z.shape[0]
        z1, z2 = torch.chunk(z, 2, dim=-1)
        if mask is not None:
            assert mask.shape[0] == 1 or mask.shape[0] == seq_len
            assert mask.shape[1] == seq_len
            assert mask.shape[2] == 1
            mask = mask[:, :, 0]
        z2 = self.norm(z2)
        weight = self.weight[:seq_len, :seq_len]
        if mask is not None:
            weight = weight * mask
        z2 = torch.einsum('ij,jbd->ibd', weight, z2) + self.bias[:seq_len,
            None, None]
        return z1 * z2


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_z': 4, 'seq_len': 4}]
