import torch
import torch.nn as nn


class FastAttention(nn.Module):
    """ wuch15's Fastformer Attention module (Official) """

    def __init__(self, dim, dim_head, heads, dropout=0.1, initializer_range
        =0.02):
        super(FastAttention, self).__init__()
        self.initializer_range = initializer_range
        if dim % dim_head != 0:
            raise ValueError(
                'The hidden size (%d) is not a multiple of the number of attention heads (%d)'
                 % (dim, dim_head))
        self.attention_head_size = int(dim / dim_head)
        self.num_attention_heads = dim_head
        self.all_head_size = (self.num_attention_heads * self.
            attention_head_size)
        self.input_dim = dim
        self.query = nn.Linear(self.input_dim, self.all_head_size)
        self.to_q_attn_logits = nn.Linear(self.all_head_size, self.
            num_attention_heads)
        self.key = nn.Linear(self.input_dim, self.all_head_size)
        self.to_k_attn_logits = nn.Linear(self.all_head_size, self.
            num_attention_heads)
        self.transform = nn.Linear(self.all_head_size, self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)
        self.apply(self.init_weights)
        self.dropout = nn.Dropout(dropout)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.
            attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, mask):
        """
        hidden_states -- [B, T, H]
        mask -- [B, T]
        """
        mask = mask.unsqueeze(1)
        mask = mask
        mask = (1.0 - mask) * -10000.0
        _batch_size, seq_len, _ = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        query_for_score = self.to_q_attn_logits(mixed_query_layer).transpose(
            1, 2) / self.attention_head_size ** 0.5
        query_for_score += mask
        query_weight = self.softmax(query_for_score).unsqueeze(2)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        pooled_query = torch.matmul(query_weight, query_layer).transpose(1, 2
            ).view(-1, 1, self.num_attention_heads * self.attention_head_size)
        pooled_query_repeat = pooled_query.repeat(1, seq_len, 1)
        mixed_query_key_layer = mixed_key_layer * pooled_query_repeat
        query_key_score = (self.to_k_attn_logits(mixed_query_key_layer) / 
            self.attention_head_size ** 0.5).transpose(1, 2)
        query_key_score += mask
        query_key_weight = self.softmax(query_key_score).unsqueeze(2)
        key_layer = self.transpose_for_scores(mixed_query_key_layer)
        pooled_key = torch.matmul(query_key_weight, key_layer)
        weighted_value = (pooled_key * query_layer).transpose(1, 2)
        weighted_value = weighted_value.reshape(weighted_value.size()[:-2] +
            (self.num_attention_heads * self.attention_head_size,))
        weighted_value = self.transform(weighted_value) + mixed_query_layer
        return self.dropout(weighted_value)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'dim_head': 4, 'heads': 4}]
