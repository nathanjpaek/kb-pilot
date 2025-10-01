import math
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torch import nn


def softmax(x):
    if x.dim() == 3:
        return F.softmax(x.transpose(0, 2)).transpose(0, 2)
    return F.softmax(x)


def gumbel_softmax(input, beta=0.5, tau=1.0):
    noise = input.data.new(*input.size()).uniform_()
    noise.add_(TINY).log_().neg_().add_(TINY).log_().neg_()
    return softmax((input + beta * Variable(noise)) / tau)


def matmul(x, y):
    if x.dim() == y.dim():
        return x @ y
    if x.dim() == y.dim() - 1:
        return (x.unsqueeze(-2) @ y).squeeze(-2)
    return (x @ y.unsqueeze(-1)).squeeze(-1)


def mask(targets, out, input_mask=None, return_mask=False):
    if input_mask is None:
        input_mask = targets != 1
    out_mask = input_mask.unsqueeze(-1).expand_as(out)
    if return_mask:
        return targets[input_mask], out[out_mask].view(-1, out.size(-1)
            ), the_mask
    return targets[input_mask], out[out_mask].view(-1, out.size(-1))


class Linear(nn.Linear):

    def forward(self, x):
        size = x.size()
        return super().forward(x.contiguous().view(-1, size[-1])).view(*
            size[:-1], -1)


class Attention(nn.Module):

    def __init__(self, d_key, drop_ratio, causal, diag=False, window=-1,
        noisy=False):
        super().__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(drop_ratio)
        self.causal = causal
        self.diag = diag
        self.window = window
        self.noisy = noisy

    def forward(self, query, key, value=None, mask=None, feedback=None,
        beta=0, tau=1, weights=None):
        dot_products = matmul(query, key.transpose(1, 2))
        if weights is not None:
            dot_products = dot_products + weights
        if query.dim() == 3 and self.causal and query.size(1) == key.size(1):
            tri = key.data.new(key.size(1), key.size(1)).fill_(1).triu(1) * INF
            dot_products.data.sub_(tri.unsqueeze(0))
        if self.window > 0:
            window_mask = key.data.new(key.size(1), key.size(1)).fill_(1)
            window_mask = (window_mask.triu(self.window + 1) + window_mask.
                tril(-self.window - 1)) * INF
            dot_products.data.sub_(window_mask.unsqueeze(0))
        if self.diag:
            inds = torch.arange(0, key.size(1)).long().view(1, 1, -1)
            if key.is_cuda:
                inds = inds
            dot_products.data.scatter_(1, inds.expand(dot_products.size(0),
                1, inds.size(-1)), -INF)
        if mask is not None:
            if dot_products.dim() == 2:
                dot_products.data -= (1 - mask) * INF
            else:
                dot_products.data -= (1 - mask[:, None, :]) * INF
        if value is None:
            return dot_products
        logits = dot_products / self.scale
        if not self.noisy:
            probs = softmax(logits)
        else:
            probs = gumbel_softmax(logits, beta=beta, tau=tau)
        if feedback is not None:
            feedback.append(probs.contiguous())
        return matmul(self.dropout(probs), value)


class MoEHead(nn.Module):

    def __init__(self, d_key, d_value, n_heads, drop_ratio, causal=False,
        diag=False, window=-1, noisy=False, use_wo=True):
        super().__init__()
        self.attention = Attention(d_key, drop_ratio, causal=causal, diag=
            diag, window=window, noisy=noisy)
        self.wq = Linear(d_key, d_key, bias=use_wo)
        self.wk = Linear(d_key, d_key, bias=use_wo)
        self.wv = Linear(d_value, d_value, bias=use_wo)
        self.wo = Linear(d_value, d_key, bias=use_wo)
        self.gate = Linear(d_value // n_heads, 1)
        self.use_wo = use_wo
        self.n_heads = n_heads

    def forward(self, query, key, inputs, mask=None, feedback=None, weights
        =None, beta=0, tau=1):
        query, key, value = self.wq(query), self.wk(key), self.wv(inputs)
        B, Tq, D = query.size()
        _, Tk, _ = key.size()
        N = self.n_heads
        probs = []
        query, key, value = (x.contiguous().view(B, -1, N, D // N).
            transpose(2, 1).contiguous().view(B * N, -1, D // N) for x in (
            query, key, value))
        if mask is not None:
            mask = mask[:, None, :].expand(B, N, Tk).contiguous().view(B *
                N, -1)
        probs = self.attention(query, key, None, mask, probs, beta, tau,
            weights)
        mix = matmul(self.attention.dropout(probs), value).contiguous().view(B,
            N, -1, D // N).transpose(2, 1).contiguous()
        mix = softmax(self.gate(mix))
        probs = (probs.contiguous().view(B, N, Tq, Tk).transpose(2, 1) * mix
            ).sum(-2)
        outputs = matmul(probs, inputs)
        return self.wo(outputs)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'d_key': 4, 'd_value': 4, 'n_heads': 4, 'drop_ratio': 0.5}]
