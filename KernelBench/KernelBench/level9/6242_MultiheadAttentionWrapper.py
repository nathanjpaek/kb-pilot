import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.optim.lr_scheduler import *
import torch.utils.data
import torch.onnx.operators
import torch.optim
import torch.optim.lr_scheduler


def linear(x):
    return x


def activation(func_a):
    """Activation function wrapper
    """
    try:
        f = eval(func_a)
    except:
        f = linear
    return f


class DropoutWrapper(nn.Module):
    """
    This is a dropout wrapper which supports the fix mask dropout
    """

    def __init__(self, dropout_p=0, enable_vbp=True):
        super(DropoutWrapper, self).__init__()
        """variational dropout means fix dropout mask
        ref: https://discuss.pytorch.org/t/dropout-for-rnns/633/11
        """
        self.enable_variational_dropout = enable_vbp
        self.dropout_p = dropout_p

    def forward(self, x):
        """
            :param x: batch * len * input_size
        """
        if self.training is False or self.dropout_p == 0:
            return x
        if len(x.size()) == 3:
            mask = 1.0 / (1 - self.dropout_p) * torch.bernoulli((1 - self.
                dropout_p) * (x.data.new(x.size(0), x.size(2)).zero_() + 1))
            mask.requires_grad = False
            return mask.unsqueeze(1).expand_as(x) * x
        else:
            return F.dropout(x, p=self.dropout_p, training=self.training)


class MultiheadAttentionWrapper(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, query_dim, key_dim, value_dim, prefix='attention',
        opt={}, dropout=None):
        super().__init__()
        self.prefix = prefix
        self.num_heads = opt.get('{}_head'.format(self.prefix), 1)
        self.dropout = DropoutWrapper(opt.get('{}_dropout'.format(self.
            prefix), 0)) if dropout is None else dropout
        self.qkv_dim = [query_dim, key_dim, value_dim]
        assert query_dim == key_dim, 'query dim must equal with key dim'
        self.hidden_size = opt.get('{}_hidden_size'.format(self.prefix), 64)
        self.proj_on = opt.get('{}_proj_on'.format(prefix), False)
        self.share = opt.get('{}_share'.format(self.prefix), False)
        self.layer_norm_on = opt.get('{}_norm_on'.format(self.prefix), False)
        self.scale_on = opt.get('{}_scale_on'.format(self.prefix), False)
        if self.proj_on:
            self.proj_modules = nn.ModuleList([nn.Linear(dim, self.
                hidden_size) for dim in self.qkv_dim[0:2]])
            if self.layer_norm_on:
                for proj in self.proj_modules:
                    proj = weight_norm(proj)
            if self.share and self.qkv_dim[0] == self.qkv_dim[1]:
                self.proj_modules[1] = self.proj_modules[0]
            self.f = activation(opt.get('{}_activation'.format(self.prefix),
                'relu'))
            self.qkv_head_dim = [self.hidden_size // self.num_heads] * 3
            self.qkv_head_dim[2] = value_dim // self.num_heads
            assert self.qkv_head_dim[0
                ] * self.num_heads == self.hidden_size, 'hidden size must be divisible by num_heads'
            assert self.qkv_head_dim[2
                ] * self.num_heads == value_dim, 'value size must be divisible by num_heads'
        else:
            self.qkv_head_dim = [(emb // self.num_heads) for emb in self.
                qkv_dim]
            assert self.qkv_head_dim[0] * self.num_heads == self.qkv_dim[0
                ], 'query size must be divisible by num_heads'
            assert self.qkv_head_dim[1] * self.num_heads == self.qkv_dim[1
                ], 'key size must be divisible by num_heads'
            assert self.qkv_head_dim[2] * self.num_heads == self.qkv_dim[2
                ], 'value size must be divisible by num_heads'
        if self.scale_on:
            self.scaling = self.qkv_head_dim[0] ** -0.5
        self.drop_diagonal = opt.get('{}_drop_diagonal'.format(self.prefix),
            False)
        self.output_size = self.qkv_dim[2]

    def forward(self, query, key, value, key_padding_mask=None):
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.qkv_dim[0]
        q, k, v = query, key, value
        if self.proj_on:
            if self.dropout:
                q, k = self.dropout(q), self.dropout(k)
            q, k = [self.f(proj(input)) for input, proj in zip([query, key],
                self.proj_modules)]
        src_len = k.size(0)
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        if self.scale_on:
            q *= self.scaling
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.
            qkv_head_dim[0]).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.
            qkv_head_dim[1]).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.
            qkv_head_dim[2]).transpose(0, 1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len,
            src_len]
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len,
                src_len)
            attn_weights = attn_weights.float().masked_fill(key_padding_mask
                .unsqueeze(1).unsqueeze(2), float('-inf')).type_as(attn_weights
                )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len,
                src_len)
        if self.drop_diagonal:
            assert attn_weights.size(1) == attn_weights.size(2)
            diag_mask = torch.diag(attn_weights.data.new(attn_weights.size(
                1)).zero_() + 1).byte().unsqueeze(0).expand_as(attn_weights)
            attn_weights.data.masked_fill_(diag_mask, -float('inf'))
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(
            attn_weights)
        attn_weights = self.dropout(attn_weights)
        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.
            qkv_head_dim[2]]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        attn = attn.transpose(0, 1)
        return attn


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'query_dim': 4, 'key_dim': 4, 'value_dim': 4}]
