import torch
import torch.nn as nn


class DenseSynthesizer(nn.Module):

    def __init__(self, head_dim, n_heads, n_tokens, big=True):
        super().__init__()
        h = max(head_dim, n_tokens) if big else min(head_dim, n_tokens)
        w1 = torch.empty(n_heads, head_dim, h)
        b1 = torch.empty(n_heads, h)
        w2 = torch.empty(n_heads, h, n_tokens)
        b2 = torch.empty(n_heads, n_tokens)
        nn.init.kaiming_uniform_(w1)
        nn.init.kaiming_uniform_(w2)
        nn.init.zeros_(b1)
        nn.init.zeros_(b2)
        self.register_parameter('w1', nn.Parameter(w1))
        self.register_parameter('b1', nn.Parameter(b1))
        self.register_parameter('w2', nn.Parameter(w2))
        self.register_parameter('b2', nn.Parameter(b2))
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        :param x: tensor of shape (batch_size, n_tokens, n_heads, head_dim)
        :return: tensor of shape (batch_size * n_heads, n_tokens, n_tokens)
        """
        bs, l, nh, _dh = x.size()
        x = torch.einsum('ijkl,klm->ijkm', x, self.w1) + self.b1
        x = self.activation(x)
        x = torch.einsum('ijkl,klm->ijkm', x, self.w2) + self.b2
        x = x[:, :, :, :l]
        x = x.transpose(0, 3).contiguous().view(l, l, bs * nh).transpose(0, 2)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'head_dim': 4, 'n_heads': 4, 'n_tokens': 4}]
