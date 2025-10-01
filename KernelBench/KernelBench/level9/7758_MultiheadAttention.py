import torch
import torch.nn as nn


def BuildDropout(dropout_type, **kwargs):
    supported_dropouts = {'droppath': DropPath, 'dropout': nn.Dropout,
        'dropout2d': nn.Dropout2d, 'dropout3d': nn.Dropout3d}
    assert dropout_type in supported_dropouts, 'unsupport dropout type %s...' % dropout_type
    return supported_dropouts[dropout_type](**kwargs)


class DropPath(nn.Module):

    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob
    """forward"""

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = self.keep_prob + torch.rand(shape, dtype=x.dtype,
            device=x.device)
        random_tensor.floor_()
        output = x.div(self.keep_prob) * random_tensor
        return output


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dims, num_heads, attn_drop=0.0, proj_drop=0.0,
        dropout_cfg=None, batch_first=False, **kwargs):
        super(MultiheadAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
            **kwargs)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = BuildDropout(dropout_cfg['type'], **
            dropout_cfg['opts']) if dropout_cfg else nn.Identity()
    """forward"""

    def forward(self, query, key=None, value=None, identity=None, query_pos
        =None, key_pos=None, attn_mask=None, key_padding_mask=None, **kwargs):
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        out = self.attn(query=query, key=key, value=value, attn_mask=
            attn_mask, key_padding_mask=key_padding_mask)[0]
        if self.batch_first:
            out = out.transpose(0, 1)
        return identity + self.dropout_layer(self.proj_drop(out))


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'embed_dims': 4, 'num_heads': 4}]
