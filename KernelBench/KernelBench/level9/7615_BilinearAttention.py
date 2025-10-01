import torch
import torch.utils.data
from torch import nn


class BilinearAttention(nn.Module):
    """
    :param enc_dim: Scalar.
    :param dec_dim: Scalar

    """

    def __init__(self, enc_dim, dec_dim):
        super(BilinearAttention, self).__init__()
        self.W = nn.Linear(enc_dim, dec_dim)

    def forward(self, h, s):
        """
        :param h: (N, Tx, Cx) Tensor. Encoder outputs
        :param s: (N, Ty/r, Cx) Tensor. Decoder inputs (previous decoder outputs)

        Returns:
            A: (N, Ty/r, Tx) Tensor. attention
            
        """
        wh = self.W(h)
        e = torch.matmul(wh, s.transpose(1, 2))
        A = torch.softmax(e.transpose(1, 2), dim=-1)
        return A


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'enc_dim': 4, 'dec_dim': 4}]
