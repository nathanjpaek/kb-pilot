import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttn(nn.Module):

    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
        pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.k_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.v_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
        self.scale = 1 / d_head ** 0.5
        self.pre_lnorm = pre_lnorm
        for m in [self.q_net, self.k_net, self.v_net, self.o_net]:
            nn.init.normal_(m.weight, 0.0, self.scale)

    def forward(self, h, attn_mask=None, mems=None):
        bsz, qlen = h.size(0), h.size(1)
        if mems is not None:
            c = torch.cat([mems, h], dim=1)
        else:
            c = h
        c.size(1)
        if self.pre_lnorm:
            h = self.layer_norm(h)
            c = self.layer_norm(c)
        head_q = self.q_net(h).view(h.size(0), h.size(1), self.n_head, self
            .d_head)
        head_k = self.k_net(c).view(c.size(0), c.size(1), self.n_head, self
            .d_head)
        head_v = self.v_net(c).view(c.size(0), c.size(1), self.n_head, self
            .d_head)
        attn_score = torch.einsum('bind,bjnd->bijn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_((attn_mask == 0).unsqueeze(1).
                    unsqueeze(-1), _INF)
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_((attn_mask == 0).unsqueeze(-1), _INF)
        attn_prob = F.softmax(attn_score, dim=2)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.einsum('bijn,bjnd->bind', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(bsz, qlen, self.n_head * self
            .d_head)
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        output = h + attn_out
        if not self.pre_lnorm:
            output = self.layer_norm(output)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'n_head': 4, 'd_model': 4, 'd_head': 4, 'dropout': 0.5}]
