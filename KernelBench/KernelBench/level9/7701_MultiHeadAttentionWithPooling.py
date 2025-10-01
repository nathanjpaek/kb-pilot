import math
import torch
import torch.nn as nn


class kAttentionPooling(nn.Module):

    def __init__(self, seq_len, hidden_size, k_heads=5):
        super().__init__()
        self.k_heads = k_heads
        self.theta_k = nn.Parameter(torch.randn([hidden_size, k_heads]))

    def forward(self, input_tensor):
        attention_matrix = torch.matmul(input_tensor, self.theta_k)
        attention_matrix = nn.Softmax(dim=-2)(attention_matrix)
        pooling_result = torch.einsum('nij, nik -> nkj', input_tensor,
            attention_matrix)
        return pooling_result


class MultiHeadAttentionWithPooling(nn.Module):

    def __init__(self, n_heads, k_heads, hidden_size, seq_len,
        hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(MultiHeadAttentionWithPooling, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                'The hidden size (%d) is not a multiple of the number of attention heads (%d)'
                 % (hidden_size, n_heads))
        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = (self.num_attention_heads * self.
            attention_head_size)
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.attpooling_key = kAttentionPooling(seq_len, hidden_size, k_heads)
        self.attpooling_value = kAttentionPooling(seq_len, hidden_size, k_heads
            )
        self.attn_scale_factor = 2
        self.pos_q_linear = nn.Linear(hidden_size, self.all_head_size)
        self.pos_k_linear = nn.Linear(hidden_size, self.all_head_size)
        self.pos_scaling = float(self.attention_head_size * self.
            attn_scale_factor) ** -0.5
        self.pos_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.
            attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, pos_emb):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(self.attpooling_key(
            mixed_key_layer))
        value_layer = self.transpose_for_scores(self.attpooling_value(
            mixed_value_layer))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
            -2))
        attention_scores = attention_scores / math.sqrt(self.
            attention_head_size)
        attention_probs = nn.Softmax(dim=-2)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer_item = torch.matmul(attention_probs, value_layer)
        value_layer_pos = self.transpose_for_scores(mixed_value_layer)
        pos_emb = self.pos_ln(pos_emb)
        pos_query_layer = self.transpose_for_scores(self.pos_q_linear(pos_emb)
            ) * self.pos_scaling
        pos_key_layer = self.transpose_for_scores(self.pos_k_linear(pos_emb))
        abs_pos_bias = torch.matmul(pos_query_layer, pos_key_layer.
            transpose(-1, -2))
        abs_pos_bias = abs_pos_bias / math.sqrt(self.attention_head_size)
        abs_pos_bias = nn.Softmax(dim=-2)(abs_pos_bias)
        context_layer_pos = torch.matmul(abs_pos_bias, value_layer_pos)
        context_layer = context_layer_item + context_layer_pos
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.
            all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'n_heads': 4, 'k_heads': 4, 'hidden_size': 4, 'seq_len': 4,
        'hidden_dropout_prob': 0.5, 'attn_dropout_prob': 0.5,
        'layer_norm_eps': 1}]
