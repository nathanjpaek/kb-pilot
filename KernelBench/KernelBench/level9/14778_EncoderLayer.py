from _paritybench_helpers import _mock_config
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


class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-06):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class ResidualBlock(nn.Module):

    def __init__(self, layer, d_model, drop_ratio, pos=0):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm = LayerNorm(d_model)
        self.pos = pos

    def forward(self, *x):
        return self.layernorm(x[self.pos] + self.dropout(self.layer(*x)))


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


class MultiHead2(nn.Module):

    def __init__(self, d_key, d_value, n_heads, drop_ratio, causal=False,
        diag=False, window=-1, noisy=False, use_wo=True):
        super().__init__()
        self.attention = Attention(d_key, drop_ratio, causal=causal, diag=
            diag, window=window, noisy=noisy)
        self.wq = Linear(d_key, d_key, bias=use_wo)
        self.wk = Linear(d_key, d_key, bias=use_wo)
        self.wv = Linear(d_value, d_value, bias=use_wo)
        if use_wo:
            self.wo = Linear(d_value, d_key, bias=use_wo)
        self.use_wo = use_wo
        self.n_heads = n_heads

    def forward(self, query, key, value, mask=None, feedback=None, weights=
        None, beta=0, tau=1):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)
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
        outputs = self.attention(query, key, value, mask, probs, beta, tau,
            weights)
        outputs = outputs.contiguous().view(B, N, -1, D // N).transpose(2, 1
            ).contiguous().view(B, -1, D)
        if feedback is not None:
            feedback.append(probs[0].view(B, N, Tq, Tk))
        if self.use_wo:
            return self.wo(outputs)
        return outputs


class FeedForward(nn.Module):

    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.linear1 = Linear(d_model, d_hidden)
        self.linear2 = Linear(d_hidden, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class EncoderLayer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.selfattn = ResidualBlock(MultiHead2(args.d_model, args.d_model,
            args.n_heads, args.drop_ratio, use_wo=args.use_wo), args.
            d_model, args.drop_ratio)
        self.feedforward = ResidualBlock(FeedForward(args.d_model, args.
            d_hidden), args.d_model, args.drop_ratio)

    def forward(self, x, mask=None):
        return self.feedforward(self.selfattn(x, x, x, mask))


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'args': _mock_config(d_model=4, n_heads=4, drop_ratio=0.5,
        use_wo=4, d_hidden=4)}]
