from _paritybench_helpers import _mock_config
import math
import torch
import torch.utils.checkpoint
from torch import nn


class AlbertAttentionWithoutSkipConnection(nn.Module):

    def __init__(self, config):
        super().__init__()
        if (config.hidden_size % config.num_attention_heads != 0 and not
            hasattr(config, 'embedding_size')):
            raise ValueError(
                'The hidden size (%d) is not a multiple of the number of attention heads (%d)'
                 % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = (config.hidden_size // config.
            num_attention_heads)
        self.all_head_size = (self.num_attention_heads * self.
            attention_head_size)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob
            )
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.
            layer_norm_eps)
        self.pruned_heads = set()
        self.position_embedding_type = getattr(config,
            'position_embedding_type', 'absolute')
        if (self.position_embedding_type == 'relative_key' or self.
            position_embedding_type == 'relative_key_query'):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.
                max_position_embeddings - 1, self.attention_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.
            attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
        output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
            -2))
        attention_scores = attention_scores / math.sqrt(self.
            attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        if (self.position_embedding_type == 'relative_key' or self.
            position_embedding_type == 'relative_key_query'):
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long,
                device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long,
                device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.
                max_position_embeddings - 1)
            positional_embedding = positional_embedding
            if self.position_embedding_type == 'relative_key':
                relative_position_scores = torch.einsum('bhld,lrd->bhlr',
                    query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == 'relative_key_query':
                relative_position_scores_query = torch.einsum('bhld,lrd->bhlr',
                    query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum('bhrd,lrd->bhlr',
                    key_layer, positional_embedding)
                attention_scores = (attention_scores +
                    relative_position_scores_query +
                    relative_position_scores_key)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attention_dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        w = self.dense.weight.t().view(self.num_attention_heads, self.
            attention_head_size, self.hidden_size)
        b = self.dense.bias
        projected_context_layer = torch.einsum('bfnd,ndh->bfh',
            context_layer, w) + b
        projected_context_layer_dropout = self.output_dropout(
            projected_context_layer)
        layernormed_context_layer = self.LayerNorm(
            projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs
            ) if output_attentions else (layernormed_context_layer,)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4, num_attention_heads=
        4, attention_probs_dropout_prob=0.5, hidden_dropout_prob=0.5,
        layer_norm_eps=1, position_embedding_type=4)}]
