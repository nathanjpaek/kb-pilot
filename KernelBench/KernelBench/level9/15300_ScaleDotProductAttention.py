import math
import torch
from torch import nn


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        _batch_size, _head, _length, d_tensor = k.size()
        k_t = k.transpose(2, 3)
        score = q @ k_t / math.sqrt(d_tensor)
        if mask is not None:
            score = score.masked_fill(mask == 0, -e)
        score = self.softmax(score)
        v = score @ v
        return v, score


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
