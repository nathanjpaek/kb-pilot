import torch
import torch.nn as nn


class SinusoidPositionalEmbedding(nn.Module):

    def forward(self, x):
        seq_len, n_model = x[0].shape
        pos = x.new_tensor(range(seq_len)).unsqueeze(-1) / 10000 ** (x.
            new_tensor(range(n_model)) // 2 * 2 / n_model)
        pos[:, 0::2], pos[:, 1::2] = pos[:, 0::2].sin(), pos[:, 1::2].cos()
        return pos


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
