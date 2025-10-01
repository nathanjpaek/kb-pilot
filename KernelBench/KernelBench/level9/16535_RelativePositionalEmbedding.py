import torch
import torch.nn as nn


class RelativePositionalEmbedding(nn.Module):

    def __init__(self, n_model, max_len=1024):
        super().__init__()
        self.embed = nn.Embedding(max_len, n_model)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        w = self.embed.weight
        max_len, n_model = w.shape
        pos = torch.cat((w.new_tensor(range(-max_len // 2, 0)), w.
            new_tensor(range(max_len // 2))))
        w = pos.unsqueeze(-1) / 10000 ** (w.new_tensor(range(n_model)) // 2 *
            2 / n_model)
        w[:, 0::2], w[:, 1::2] = w[:, 0::2].sin(), w[:, 1::2].cos()
        self.embed.weight.copy_(w)

    def forward(self, x):
        pos = x.new_tensor(range(x.shape[1])).long()
        offset = sum(divmod(self.embed.weight.shape[0], 2))
        return self.embed(pos - pos.unsqueeze(-1) + offset)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_model': 4}]
