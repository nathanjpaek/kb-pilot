import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import LayerNorm


class MultiheadAttention(nn.Module):
    """Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        	ext{MultiHead}(Q, K, V) = 	ext{Concat}(head_1,\\dots,head_h)W^O
        	ext{where} head_i = 	ext{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model
        num_heads: parallel attention layers, or heads

    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True,
        add_bias_kv=False, add_zero_attn=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.scaling = self.head_dim ** -0.5
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()
        self.conv1 = torch.nn.Conv2d(in_channels=embed_dim, out_channels=
            embed_dim, kernel_size=(1, 1))
        self.conv2 = torch.nn.Conv2d(in_channels=embed_dim, out_channels=
            embed_dim, kernel_size=(1, 1))
        self.conv3 = torch.nn.Conv2d(in_channels=embed_dim, out_channels=
            embed_dim, kernel_size=(1, 1))

    def _reset_parameters(self):
        xavier_uniform_(self.out_proj.weight)

    def forward(self, query, key, value, key_padding_mask=None,
        incremental_state=None, attn_mask=None):
        """
        Inputs of forward function
            query: [target length, batch size, embed dim]
            key: [sequence length, batch size, embed dim]
            value: [sequence length, batch size, embed dim]
            key_padding_mask: if True, mask padding based on batch size
            incremental_state: if provided, previous time steps are cashed
            need_weights: output attn_output_weights
            static_kv: key and value are static

        Outputs of forward function
            attn_output: [target length, batch size, embed dim]
            attn_output_weights: [batch size, target length, sequence length]
        """
        q_shape = query.shape
        src_shape = key.shape
        q = self._in_proj_q(query)
        k = self._in_proj_k(key)
        v = self._in_proj_v(value)
        q *= self.scaling
        q = torch.reshape(q, (q_shape[0], q_shape[1], self.num_heads, self.
            head_dim))
        q = q.permute(1, 2, 0, 3)
        k = torch.reshape(k, (src_shape[0], q_shape[1], self.num_heads,
            self.head_dim))
        k = k.permute(1, 2, 0, 3)
        v = torch.reshape(v, (src_shape[0], q_shape[1], self.num_heads,
            self.head_dim))
        v = v.permute(1, 2, 0, 3)
        if key_padding_mask is not None:
            assert key_padding_mask.shape[0] == q_shape[1]
            assert key_padding_mask.shape[1] == src_shape[0]
        attn_output_weights = torch.matmul(q, k.permute(0, 1, 3, 2))
        if attn_mask is not None:
            attn_mask = torch.unsqueeze(torch.unsqueeze(attn_mask, 0), 0)
            attn_output_weights += attn_mask
        if key_padding_mask is not None:
            attn_output_weights = torch.reshape(attn_output_weights, [
                q_shape[1], self.num_heads, q_shape[0], src_shape[0]])
            key = torch.unsqueeze(torch.unsqueeze(key_padding_mask, 1), 2)
            key = key.type(torch.float32)
            y = torch.full(size=key.shape, fill_value=float('-Inf'), dtype=
                torch.float32)
            y = torch.where(key == 0.0, key, y)
            attn_output_weights += y
        attn_output_weights = F.softmax(attn_output_weights.type(torch.
            float32), dim=-1, dtype=torch.float32 if attn_output_weights.
            dtype == torch.float16 else attn_output_weights.dtype)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout,
            training=self.training)
        attn_output = torch.matmul(attn_output_weights, v)
        attn_output = torch.reshape(attn_output.permute(2, 0, 1, 3), [
            q_shape[0], q_shape[1], self.embed_dim])
        attn_output = self.out_proj(attn_output)
        return attn_output

    def _in_proj_q(self, query):
        query = query.permute(1, 2, 0)
        query = torch.unsqueeze(query, dim=2)
        res = self.conv1(query)
        res = torch.squeeze(res, dim=2)
        res = res.permute(2, 0, 1)
        return res

    def _in_proj_k(self, key):
        key = key.permute(1, 2, 0)
        key = torch.unsqueeze(key, dim=2)
        res = self.conv2(key)
        res = torch.squeeze(res, dim=2)
        res = res.permute(2, 0, 1)
        return res

    def _in_proj_v(self, value):
        value = value.permute(1, 2, 0)
        value = torch.unsqueeze(value, dim=2)
        res = self.conv3(value)
        res = torch.squeeze(res, dim=2)
        res = res.permute(2, 0, 1)
        return res


class TransformerDecoderLayer(nn.Module):
    """TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).

    """

    def __init__(self, d_model, nhead, dim_feedforward=2048,
        attention_dropout_rate=0.0, residual_dropout_rate=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=
            attention_dropout_rate)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=
            attention_dropout_rate)
        self.conv1 = nn.Conv2d(in_channels=d_model, out_channels=
            dim_feedforward, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=dim_feedforward, out_channels=
            d_model, kernel_size=(1, 1))
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(residual_dropout_rate)
        self.dropout2 = Dropout(residual_dropout_rate)
        self.dropout3 = Dropout(residual_dropout_rate)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=
            memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt = tgt.permute(1, 2, 0)
        tgt = torch.unsqueeze(tgt, 2)
        tgt2 = self.conv2(F.relu(self.conv1(tgt)))
        tgt2 = torch.squeeze(tgt2, 2)
        tgt2 = tgt2.permute(2, 0, 1)
        tgt = torch.squeeze(tgt, 2)
        tgt = tgt.permute(2, 0, 1)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'nhead': 4}]
