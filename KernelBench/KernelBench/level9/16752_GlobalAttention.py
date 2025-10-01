import torch
import torch.nn as nn


def aeq(*args):
    base = args[0]
    for a in args[1:]:
        assert a == base, str(args)


class Bottle(nn.Module):

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.contiguous().view(size[0], size[1], -1)


class BottleLinear(Bottle, nn.Linear):
    pass


class GlobalAttention(nn.Module):
    """
    Luong Attention.

    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.


        H_1 H_2 H_3 ... H_n
          q   q   q       q
            |  |   |       |
              \\ |   |      /
                      .....
                  \\   |  /
                      a

    Constructs a unit mapping.
    $$(H_1 + H_n, q) => (a)$$
    Where H is of `batch x n x dim` and q is of `batch x dim`.

    Luong Attention (dot, general):
    The full function is
    $$	anh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.

    * dot: $$score(h_t,{\\overline{h}}_s) = h_t^T{\\overline{h}}_s$$
    * general: $$score(h_t,{\\overline{h}}_s) = h_t^T W_a {\\overline{h}}_s$$

    Bahdanau Attention (mlp):
    $$c = \\sum_{j=1}^{SeqLength}_jh_j$$.
    The Alignment-function $$a$$ computes an alignment as:
    $$a_j = softmax(v_a^T 	anh(W_a q + U_a h_j) )$$.

    """

    def __init__(self, dim, coverage=False, attn_type='dot'):
        super(GlobalAttention, self).__init__()
        self.dim = dim
        self.attn_type = attn_type
        assert self.attn_type in ['dot', 'general', 'mlp'
            ], 'Please select a valid attention type.'
        if self.attn_type == 'general':
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == 'mlp':
            self.linear_context = BottleLinear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = BottleLinear(dim, 1, bias=False)
        out_bias = self.attn_type == 'mlp'
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)
        self.sm = nn.Softmax()
        self.tanh = nn.Tanh()
        self.mask = None
        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)

    def applyMask(self, mask):
        self.mask = mask

    def score(self, h_t, h_s):
        """
        h_t (FloatTensor): batch x dim
        h_s (FloatTensor): batch x src_len x dim
        returns scores (FloatTensor): batch x src_len:
            raw attention scores for each src index
        """
        src_batch, _, src_dim = h_s.size()
        tgt_batch, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)
        if self.attn_type in ['general', 'dot']:
            if self.attn_type == 'general':
                h_t = self.linear_in(h_t)
            return torch.bmm(h_s, h_t.unsqueeze(2)).squeeze(2)
        else:
            wq = self.linear_query(h_t).unsqueeze(1)
            uh = self.linear_context(h_s.contiguous())
            wquh = uh + wq.expand_as(uh)
            wquh = self.tanh(wquh)
            return self.v(wquh.contiguous()).squeeze(2)

    def forward(self, input, context, coverage=None):
        """
        input (FloatTensor): batch x dim: decoder's rnn's output.
        context (FloatTensor): batch x src_len x dim: src hidden states
        coverage (FloatTensor): batch x src_len
        """
        batch, sourceL, dim = context.size()
        batch_, dim_ = input.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)
        if coverage is not None:
            batch_, sourceL_ = coverage.size()
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)
        if self.mask is not None:
            beam_, batch_, sourceL_ = self.mask.size()
            aeq(batch, batch_ * beam_)
            aeq(sourceL, sourceL_)
        if coverage is not None:
            cover = coverage.view(-1).unsqueeze(1)
            context += self.linear_cover(cover).view_as(context)
            context = self.tanh(context)
        a_t = self.score(input, context)
        if self.mask is not None:
            a_t.data.masked_fill_(self.mask, -float('inf'))
        align_vector = self.sm(a_t)
        c_t = torch.bmm(align_vector.unsqueeze(1), context).squeeze(1)
        attn_h_t = self.linear_out(torch.cat([c_t, input], 1))
        if self.attn_type in ['general', 'dot']:
            attn_h_t = self.tanh(attn_h_t)
        batch_, sourceL_ = align_vector.size()
        aeq(batch, batch_)
        aeq(sourceL, sourceL_)
        batch_, dim_ = attn_h_t.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        return attn_h_t, align_vector


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
