import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """SelfAttention class"""

    def __init__(self, input_dim: 'int', da: 'int', r: 'int') ->None:
        """Instantiating SelfAttention class

        Args:
            input_dim (int): dimension of input, eg) (batch_size, seq_len, input_dim)
            da (int): the number of features in hidden layer from self-attention
            r (int): the number of aspects of self-attention
        """
        super(SelfAttention, self).__init__()
        self._ws1 = nn.Linear(input_dim, da, bias=False)
        self._ws2 = nn.Linear(da, r, bias=False)

    def forward(self, h: 'torch.Tensor') ->torch.Tensor:
        attn_mat = F.softmax(self._ws2(torch.tanh(self._ws1(h))), dim=1)
        attn_mat = attn_mat.permute(0, 2, 1)
        return attn_mat


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'da': 4, 'r': 4}]
