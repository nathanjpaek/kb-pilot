import torch
from torch import nn
import torch.nn.functional as F
import torch.optim


def get_activation_fn(name):
    """Returns a callable activation function from torch."""
    if name in (None, 'linear'):
        return lambda x: x
    elif name in ('sigmoid', 'tanh'):
        return getattr(torch, name)
    else:
        return getattr(F, name)


class DotAttention(nn.Module):
    """Attention layer with dot product."""

    def __init__(self, ctx_dim, hid_dim, att_bottleneck='ctx',
        transform_ctx=True, att_activ='tanh', temp=1.0, ctx2hid=True,
        mlp_bias=None):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.hid_dim = hid_dim
        self._ctx2hid = ctx2hid
        self.temperature = temp
        self.activ = get_activation_fn(att_activ)
        if isinstance(att_bottleneck, int):
            self.mid_dim = att_bottleneck
        else:
            self.mid_dim = getattr(self, '{}_dim'.format(att_bottleneck))
        self.hid2ctx = nn.Linear(self.hid_dim, self.mid_dim, bias=False)
        if transform_ctx or self.mid_dim != self.ctx_dim:
            self.ctx2ctx = nn.Linear(self.ctx_dim, self.mid_dim, bias=False)
        else:
            self.ctx2ctx = lambda x: x
        if self._ctx2hid:
            self.ctx2hid = nn.Linear(self.ctx_dim, self.hid_dim, bias=False)
        else:
            self.ctx2hid = lambda x: x

    def forward(self, hid, ctx, ctx_mask=None):
        """Computes attention probabilities and final context using
        decoder's hidden state and source annotations.

        Arguments:
            hid(Tensor): A set of decoder hidden states of shape `T*B*H`
                where `T` == 1, `B` is batch dim and `H` is hidden state dim.
            ctx(Tensor): A set of annotations of shape `S*B*C` where `S`
                is the source timestep dim, `B` is batch dim and `C`
                is annotation dim.
            ctx_mask(FloatTensor): A binary mask of shape `S*B` with zeroes
                in the padded positions.

        Returns:
            scores(Tensor): A tensor of shape `S*B` containing normalized
                attention scores for each position and sample.
            z_t(Tensor): A tensor of shape `B*H` containing the final
                attended context vector for this target decoding timestep.

        Notes:
            This will only work when `T==1` for now.
        """
        ctx_ = self.ctx2ctx(ctx)
        hid_ = self.hid2ctx(hid)
        scores = torch.bmm(hid_.permute(1, 0, 2), ctx_.permute(1, 2, 0)).div(
            self.temperature).squeeze(1).t()
        if ctx_mask is not None:
            scores.masked_fill_((1 - ctx_mask).byte(), -100000000.0)
        alpha = F.softmax(scores, dim=0)
        return alpha, self.ctx2hid((alpha.unsqueeze(-1) * ctx).sum(0))


class MLPAttention(DotAttention):
    """Attention layer with feed-forward layer."""

    def __init__(self, ctx_dim, hid_dim, att_bottleneck='ctx',
        transform_ctx=True, att_activ='tanh', mlp_bias=False, temp=1.0,
        ctx2hid=True):
        super().__init__(ctx_dim, hid_dim, att_bottleneck, transform_ctx,
            att_activ, temp, ctx2hid)
        if mlp_bias:
            self.bias = nn.Parameter(torch.Tensor(self.mid_dim))
            self.bias.data.zero_()
        else:
            self.register_parameter('bias', None)
        self.mlp = nn.Linear(self.mid_dim, 1, bias=False)

    def forward(self, hid, ctx, ctx_mask=None):
        """Computes attention probabilities and final context using
        decoder's hidden state and source annotations.

        Arguments:
            hid(Tensor): A set of decoder hidden states of shape `T*B*H`
                where `T` == 1, `B` is batch dim and `H` is hidden state dim.
            ctx(Tensor): A set of annotations of shape `S*B*C` where `S`
                is the source timestep dim, `B` is batch dim and `C`
                is annotation dim.
            ctx_mask(FloatTensor): A binary mask of shape `S*B` with zeroes
                in the padded positions.

        Returns:
            scores(Tensor): A tensor of shape `S*B` containing normalized
                attention scores for each position and sample.
            z_t(Tensor): A tensor of shape `B*H` containing the final
                attended context vector for this target decoding timestep.

        Notes:
            This will only work when `T==1` for now.
        """
        inner_sum = self.ctx2ctx(ctx) + self.hid2ctx(hid)
        if self.bias is not None:
            inner_sum.add_(self.bias)
        scores = self.mlp(self.activ(inner_sum)).div(self.temperature).squeeze(
            -1)
        if ctx_mask is not None:
            scores.masked_fill_((1 - ctx_mask).byte(), -100000000.0)
        alpha = F.softmax(scores, dim=0)
        return alpha, self.ctx2hid((alpha.unsqueeze(-1) * ctx).sum(0))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'ctx_dim': 4, 'hid_dim': 4}]
