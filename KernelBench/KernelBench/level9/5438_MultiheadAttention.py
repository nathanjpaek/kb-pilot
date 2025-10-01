import torch
import torch.nn as nn


class MultiheadAttention(nn.Module):
    """A warpper for torch.nn.MultiheadAttention.

    This module implements MultiheadAttention with residual connection.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads. Same as
            `nn.MultiheadAttention`.
        attn_drop (float): A Dropout layer on attn_output_weights. Default 0.0.
        proj_drop (float): The drop out rate after attention. Default 0.0.
    """

    def __init__(self, embed_dims, num_heads, attn_drop=0.0, proj_drop=0.0):
        super(MultiheadAttention, self).__init__()
        assert embed_dims % num_heads == 0, f'embed_dims must be divisible by num_heads. got {embed_dims} and {num_heads}.'
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop)
        self.dropout = nn.Dropout(proj_drop)

    def forward(self, x, key=None, value=None, residual=None, query_pos=
        None, key_pos=None, attn_mask=None, key_padding_mask=None):
        """Forward function for `MultiheadAttention`.

        Args:
            x (Tensor): The input query with shape [num_query, bs,
                embed_dims]. Same in `nn.MultiheadAttention.forward`.
            key (Tensor): The key tensor with shape [num_key, bs,
                embed_dims]. Same in `nn.MultiheadAttention.forward`.
                Default None. If None, the `query` will be used.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Default None.
                If None, the `key` will be used.
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. Default None. If not None, it will
                be added to `x` before forward function.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Default None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`.
            attn_mask (Tensor): ByteTensor mask with shape [num_query,
                num_key]. Same in `nn.MultiheadAttention.forward`.
                Default None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_key].
                Same in `nn.MultiheadAttention.forward`. Default None.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        query = x
        if key is None:
            key = query
        if value is None:
            value = key
        if residual is None:
            residual = x
        if key_pos is None:
            if query_pos is not None and key is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        out = self.attn(query, key, value=value, attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]
        return residual + self.dropout(out)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'embed_dims': 4, 'num_heads': 4}]
