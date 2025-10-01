import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len).type_as(lengths).repeat(batch_size, 1).lt(
        lengths.unsqueeze(1))


class AttnScore(nn.Module):

    def __init__(self, input_size, activation=nn.Tanh(), method='dot'):
        super(AttnScore, self).__init__()
        self.activation = activation
        self.input_size = input_size
        self.method = method
        if method == 'general':
            self.linear = nn.Linear(input_size, input_size)
            init.uniform(self.linear.weight.data, -0.005, 0.005)
        elif method == 'concat':
            self.linear_1 = nn.Linear(input_size * 2, input_size)
            self.linear_2 = nn.Linear(input_size, 1)
            init.uniform(self.linear_1.weight.data, -0.005, 0.005)
            init.uniform(self.linear_2.weight.data, -0.005, 0.005)
        elif method == 'tri_concat':
            self.linear = nn.Linear(input_size * 3, 1)
            init.uniform(self.linear.weight.data, -0.005, 0.005)

    def forward(self, h1, h2, h1_lens=None, h2_lens=None, normalize=True):
        """
        :param h1: b x m x d
        :param h2: b x n x d
        :return: attn_weights: b x 1 x m
        """
        _bsize, seq_l1, _dim = h1.size()
        _bsize, seq_l2, _dim = h2.size()
        assert h1.size(-1) == self.input_size
        assert h2.size(-1) == self.input_size
        if self.method == 'dot':
            align = h2.bmm(h1.transpose(1, 2))
        elif self.method == 'general':
            align = h2.bmm(self.linear(h1).transpose(1, 2))
        elif self.method == 'concat':
            h1 = h1.unsqueeze(1).repeat(1, seq_l2, 1, 1)
            h2 = h2.unsqueeze(2).repeat(1, 1, seq_l1, 1)
            align = self.linear_2(self.activation(self.linear_1(torch.cat([
                h1, h2], dim=3)))).squeeze(-1)
            align = F.softmax(align, dim=2)
        elif self.method == 'tri_concat':
            h1 = h1.unsqueeze(1).repeat(1, seq_l2, 1, 1)
            h2 = h2.unsqueeze(2).repeat(1, 1, seq_l1, 1)
            align = self.linear(torch.cat([h1, h2, h1 * h2], dim=3)).squeeze(-1
                )
        if h1_lens is not None:
            mask = sequence_mask(h1_lens, max_len=seq_l1).unsqueeze(1)
            align.data.masked_fill_(1 - mask, -100000000.0)
        if normalize:
            attn_weights = F.softmax(align, dim=2)
        else:
            attn_weights = F.softmax(align, dim=2)
        return attn_weights


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
