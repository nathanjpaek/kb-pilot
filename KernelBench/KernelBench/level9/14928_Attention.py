import torch
import torch.nn.functional as F
import torch.nn as nn


def dispatcher(dispatch_fn):

    def decorated(key, *args):
        if callable(key):
            return key
        if key is None:
            key = 'none'
        return dispatch_fn(key, *args)
    return decorated


def spectral_norm(module):
    """ init & apply spectral norm """
    nn.init.xavier_uniform_(module.weight, 2 ** 0.5)
    if hasattr(module, 'bias') and module.bias is not None:
        module.bias.data.zero_()
    return nn.utils.spectral_norm(module)


@dispatcher
def w_norm_dispatch(w_norm):
    return {'spectral': spectral_norm, 'none': lambda x: x}[w_norm.lower()]


def split_dim(x, dim, n_chunks):
    shape = x.shape
    assert shape[dim] % n_chunks == 0
    return x.view(*shape[:dim], n_chunks, shape[dim] // n_chunks, *shape[
        dim + 1:])


class RelativePositionalEmbedding2d(nn.Module):
    """ Learned relative positional embedding
    return Q * (R_x + R_y) for input Q and learned embedding R
    """

    def __init__(self, emb_dim, H, W, down_kv=False):
        super().__init__()
        self.H = H
        self.W = W
        self.down_kv = down_kv
        self.h_emb = nn.Embedding(H * 2 - 1, emb_dim)
        self.w_emb = nn.Embedding(W * 2 - 1, emb_dim)
        rel_y, rel_x = self.rel_grid()
        self.register_buffer('rel_y', rel_y)
        self.register_buffer('rel_x', rel_x)

    def rel_grid(self):
        y, x = torch.meshgrid(torch.arange(self.H), torch.arange(self.W))
        rel_y = y.reshape(1, -1) - y.reshape(-1, 1)
        rel_x = x.reshape(1, -1) - x.reshape(-1, 1)
        if self.down_kv:

            def down(x):
                n_q, n_k = x.shape
                x = x.view(n_q, 1, int(n_k ** 0.5), int(n_k ** 0.5))
                return (F.avg_pool2d(x.float(), 2) - 0.5).flatten(1).long()
            rel_y = down(rel_y)
            rel_x = down(rel_x)
        rel_y += self.H - 1
        rel_x += self.W - 1
        return rel_y, rel_x

    def forward(self, query):
        """
        Args:
            query: [B, n_heads, C_qk, H*W]

        return:
            [B, n_heads, H*W, H*W]
        """
        r_x = self.w_emb(self.rel_x)
        r_y = self.h_emb(self.rel_y)
        S_rel = torch.einsum('bhci,ijc->bhij', query, r_x + r_y)
        return S_rel


class Attention(nn.Module):

    def __init__(self, C_in_q, C_in_kv, C_qk, C_v, w_norm='none', scale=
        False, n_heads=1, down_kv=False, rel_pos_size=None):
        """
        Args:
            C_in_q: query source (encoder feature x)
            C_in_kv: key/value source (decoder feature y)
            C_qk: inner query/key dim, which should be same
            C_v: inner value dim, which same as output dim

            down_kv: Area attention for lightweight self-attention
                w/ mean pooling.
            rel_pos_size: height & width for relative positional embedding.
                If None or 0 is given, do not use relative positional embedding.
        """
        super().__init__()
        self.n_heads = n_heads
        self.down_kv = down_kv
        w_norm = w_norm_dispatch(w_norm)
        self.q_proj = w_norm(nn.Conv1d(C_in_q, C_qk, 1))
        self.k_proj = w_norm(nn.Conv1d(C_in_kv, C_qk, 1))
        self.v_proj = w_norm(nn.Conv1d(C_in_kv, C_v, 1))
        self.out = w_norm(nn.Conv2d(C_v, C_v, 1))
        if scale:
            self.scale = 1.0 / C_qk ** 0.5
        if rel_pos_size:
            C_h_qk = C_qk // n_heads
            self.rel_pos = RelativePositionalEmbedding2d(C_h_qk,
                rel_pos_size, rel_pos_size, down_kv=down_kv)

    def forward(self, x, y):
        """ Attend from x (decoder) to y (encoder)

        Args:
            x: decoder feature
            y: encoder feature
        """
        B, C, H, W = x.shape
        flat_x = x.flatten(start_dim=2)
        if not self.down_kv:
            flat_y = y.flatten(start_dim=2)
        else:
            y_down = F.avg_pool2d(y, 2)
            flat_y = y_down.flatten(2)
        query = self.q_proj(flat_x)
        key = self.k_proj(flat_y)
        value = self.v_proj(flat_y)
        query = split_dim(query, 1, self.n_heads)
        key = split_dim(key, 1, self.n_heads)
        value = split_dim(value, 1, self.n_heads)
        attn_score = torch.einsum('bhcq,bhck->bhqk', query, key)
        if hasattr(self, 'rel_pos'):
            attn_score += self.rel_pos(query)
        if hasattr(self, 'scale'):
            attn_score *= self.scale
        attn_w = F.softmax(attn_score, dim=-1)
        attn_out = torch.einsum('bhqk,bhck->bhcq', attn_w, value).reshape(B,
            C, H, W)
        out = self.out(attn_out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'C_in_q': 4, 'C_in_kv': 4, 'C_qk': 4, 'C_v': 4}]
