import math
import torch
from torch import nn
import torch.utils.data


def seq_mask(seq_len, max_len):
    """Create sequence mask.

    :param seq_len: list or torch.Tensor, the lengths of sequences in a batch.
    :param max_len: int, the maximum sequence length in a batch.
    :return: mask, torch.LongTensor, [batch_size, max_len]

    """
    if not isinstance(seq_len, torch.Tensor):
        seq_len = torch.LongTensor(seq_len)
    seq_len = seq_len.view(-1, 1).long()
    seq_range = torch.arange(start=0, end=max_len, dtype=torch.long, device
        =seq_len.device).view(1, -1)
    return torch.gt(seq_len, seq_range)


class DotAtte(nn.Module):

    def __init__(self, key_size, value_size):
        super(DotAtte, self).__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.scale = math.sqrt(key_size)

    def forward(self, Q, K, V, seq_mask=None):
        """

        :param Q: [batch, seq_len, key_size]
        :param K: [batch, seq_len, key_size]
        :param V: [batch, seq_len, value_size]
        :param seq_mask: [batch, seq_len]
        """
        output = torch.matmul(Q, K.transpose(1, 2)) / self.scale
        if seq_mask is not None:
            output.masked_fill_(seq_mask.lt(1), -float('inf'))
        output = nn.functional.softmax(output, dim=2)
        return torch.matmul(output, V)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'key_size': 4, 'value_size': 4}]
