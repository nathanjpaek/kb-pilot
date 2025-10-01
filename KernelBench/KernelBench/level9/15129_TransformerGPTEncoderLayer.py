import math
import torch
import torch.nn as nn
import torch.cuda
import torch.distributed


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 *
        torch.pow(x, 3))))


def generate_relative_positions_matrix(length, max_relative_positions,
    cache=False):
    """Generate the clipped relative positions matrix
       for a given length and maximum relative positions"""
    if cache:
        distance_mat = torch.arange(-length + 1, 1, 1).unsqueeze(0)
    else:
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(-1).expand(-1, length).transpose(0, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
    distance_mat_clipped = torch.clamp(distance_mat, min=-
        max_relative_positions, max=max_relative_positions)
    final_mat = distance_mat_clipped + max_relative_positions
    return final_mat


def relative_matmul(x, z, transpose):
    """Helper function for relative positions attention."""
    batch_size = x.shape[0]
    heads = x.shape[1]
    length = x.shape[2]
    x_t = x.permute(2, 0, 1, 3)
    x_t_r = x_t.reshape(length, heads * batch_size, -1)
    if transpose:
        z_t = z.transpose(1, 2)
        x_tz_matmul = torch.matmul(x_t_r, z_t)
    else:
        x_tz_matmul = torch.matmul(x_t_r, z)
    x_tz_matmul_r = x_tz_matmul.reshape(length, batch_size, heads, -1)
    x_tz_matmul_r_t = x_tz_matmul_r.permute(1, 2, 0, 3)
    return x_tz_matmul_r_t


class MLP(nn.Module):

    def __init__(self, n_embd, n_state, dropout):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(n_embd, n_state)
        self.c_proj = nn.Linear(n_state, n_embd)
        self.act = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        self.c_fc.weight.data.normal_(std=0.02)
        self.c_fc.bias.data.zero_()
        self.c_proj.weight.data.normal_(std=0.02)
        self.c_proj.bias.data.zero_()

    def forward(self, x):
        """
            x is input, [T, B, n_state]
        """
        h = self.dropout_1(self.act(self.c_fc(x)))
        h2 = self.dropout_2(self.c_proj(h))
        return h2


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention module from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1,
        max_relative_positions=0):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count
        self.linear_keys = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim, head_count * self.
            dim_per_head)
        self.linear_query = nn.Linear(model_dim, head_count * self.dim_per_head
            )
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)
        self.max_relative_positions = max_relative_positions
        if max_relative_positions > 0:
            vocab_size = max_relative_positions * 2 + 1
            self.relative_positions_embeddings = nn.Embedding(vocab_size,
                self.dim_per_head)

    def forward(self, key, value, query, mask=None, layer_cache=None, type=None
        ):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask indicating which keys have
               non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):

           * output context vectors ``(batch, query_len, dim)``
           * one of the attention vectors ``(batch, query_len, key_len)``
        """
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(
                1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous().view(batch_size, -1, 
                head_count * dim_per_head)
        if layer_cache is not None:
            if type == 'self':
                query, key, value = self.linear_query(query), self.linear_keys(
                    query), self.linear_values(query)
                key = shape(key)
                value = shape(value)
                if layer_cache['self_keys'] is not None:
                    key = torch.cat((layer_cache['self_keys'], key), dim=2)
                if layer_cache['self_values'] is not None:
                    value = torch.cat((layer_cache['self_values'], value),
                        dim=2)
                layer_cache['self_keys'] = key
                layer_cache['self_values'] = value
            elif type == 'context':
                query = self.linear_query(query)
                if layer_cache['memory_keys'] is None:
                    key, value = self.linear_keys(key), self.linear_values(
                        value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache['memory_keys'], layer_cache[
                        'memory_values']
                layer_cache['memory_keys'] = key
                layer_cache['memory_values'] = value
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)
        if self.max_relative_positions > 0 and type == 'self':
            key_len = key.size(2)
            relative_positions_matrix = generate_relative_positions_matrix(
                key_len, self.max_relative_positions, cache=True if 
                layer_cache is not None else False)
            relations_keys = self.relative_positions_embeddings(
                relative_positions_matrix)
            relations_values = self.relative_positions_embeddings(
                relative_positions_matrix)
        query = shape(query)
        key_len = key.size(2)
        query_len = query.size(2)
        query = query / math.sqrt(dim_per_head)
        query_key = torch.matmul(query, key.transpose(2, 3))
        if self.max_relative_positions > 0 and type == 'self':
            scores = query_key + relative_matmul(query, relations_keys, True)
        else:
            scores = query_key
        scores = scores.float()
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, -1e+18)
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context_original = torch.matmul(drop_attn, value)
        if self.max_relative_positions > 0 and type == 'self':
            context = unshape(context_original + relative_matmul(drop_attn,
                relations_values, False))
        else:
            context = unshape(context_original)
        output = self.final_linear(context)
        top_attn = attn.view(batch_size, head_count, query_len, key_len)[:,
            0, :, :].contiguous()
        return output, top_attn


class TransformerGPTEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout, attn_dropout,
        max_relative_positions=0):
        super(TransformerGPTEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(heads, d_model, dropout=
            attn_dropout, max_relative_positions=max_relative_positions)
        self.feed_forward = MLP(d_model, d_model * 4, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-05)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-05)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, src_len, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        dec_mask = None
        src_len = mask.size(-1)
        future_mask = torch.ones([src_len, src_len], device=mask.device,
            dtype=torch.uint8)
        future_mask = future_mask.triu_(1).view(1, src_len, src_len)
        dec_mask = torch.gt(mask + future_mask, 0)
        input_norm = self.layer_norm_1(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
            mask=dec_mask, type='self')
        context = self.dropout(context) + inputs
        context_norm = self.layer_norm_2(context)
        output = self.feed_forward(context_norm)
        output = output + context
        return output


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'heads': 4, 'd_ff': 4, 'dropout': 0.5,
        'attn_dropout': 0.5}]
