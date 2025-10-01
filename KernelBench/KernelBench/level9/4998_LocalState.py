import math
import torch
from torch import nn


class LocalState(nn.Module):
    """Local state allows to have attention based only on data (no positional embedding),
    but while setting a constraint on the time window (e.g. decaying penalty term).

    Also a failed experiments with trying to provide some frequency based attention.
    """

    def __init__(self, channels: 'int', heads: 'int'=4, nfreqs: 'int'=0,
        ndecay: 'int'=4):
        super().__init__()
        assert channels % heads == 0, (channels, heads)
        self.heads = heads
        self.nfreqs = nfreqs
        self.ndecay = ndecay
        self.content = nn.Conv1d(channels, channels, 1)
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)
        if nfreqs:
            self.query_freqs = nn.Conv1d(channels, heads * nfreqs, 1)
        if ndecay:
            self.query_decay = nn.Conv1d(channels, heads * ndecay, 1)
            self.query_decay.weight.data *= 0.01
            assert self.query_decay.bias is not None
            self.query_decay.bias.data[:] = -2
        self.proj = nn.Conv1d(channels + heads * nfreqs, channels, 1)

    def forward(self, x):
        B, _C, T = x.shape
        heads = self.heads
        indexes = torch.arange(T, device=x.device, dtype=x.dtype)
        delta = indexes[:, None] - indexes[None, :]
        queries = self.query(x).view(B, heads, -1, T)
        keys = self.key(x).view(B, heads, -1, T)
        dots = torch.einsum('bhct,bhcs->bhts', keys, queries)
        dots /= keys.shape[2] ** 0.5
        if self.nfreqs:
            periods = torch.arange(1, self.nfreqs + 1, device=x.device,
                dtype=x.dtype)
            freq_kernel = torch.cos(2 * math.pi * delta / periods.view(-1, 
                1, 1))
            freq_q = self.query_freqs(x).view(B, heads, -1, T
                ) / self.nfreqs ** 0.5
            dots += torch.einsum('fts,bhfs->bhts', freq_kernel, freq_q)
        if self.ndecay:
            decays = torch.arange(1, self.ndecay + 1, device=x.device,
                dtype=x.dtype)
            decay_q = self.query_decay(x).view(B, heads, -1, T)
            decay_q = torch.sigmoid(decay_q) / 2
            decay_kernel = -decays.view(-1, 1, 1) * delta.abs(
                ) / self.ndecay ** 0.5
            dots += torch.einsum('fts,bhfs->bhts', decay_kernel, decay_q)
        dots.masked_fill_(torch.eye(T, device=dots.device, dtype=torch.bool
            ), -100)
        weights = torch.softmax(dots, dim=2)
        content = self.content(x).view(B, heads, -1, T)
        result = torch.einsum('bhts,bhct->bhcs', weights, content)
        if self.nfreqs:
            time_sig = torch.einsum('bhts,fts->bhfs', weights, freq_kernel)
            result = torch.cat([result, time_sig], 2)
        result = result.reshape(B, -1, T)
        return x + self.proj(result)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]
