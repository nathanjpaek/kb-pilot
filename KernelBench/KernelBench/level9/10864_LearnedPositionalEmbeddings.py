from torch.nn import Module
import torch
from torch import nn
import torch.utils.data
import torch.nn.functional
import torch.autograd


class LearnedPositionalEmbeddings(Module):
    """
    <a id="LearnedPositionalEmbeddings">
    ## Add parameterized positional encodings
    </a>

    This adds learned positional embeddings to patch embeddings.
    """

    def __init__(self, d_model: 'int', max_len: 'int'=5000):
        """
        * `d_model` is the transformer embeddings size
        * `max_len` is the maximum number of patches
        """
        super().__init__()
        self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1,
            d_model), requires_grad=True)

    def forward(self, x: 'torch.Tensor'):
        """
        * `x` is the patch embeddings of shape `[patches, batch_size, d_model]`
        """
        pe = self.positional_encodings[x.shape[0]]
        return x + pe


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4}]
