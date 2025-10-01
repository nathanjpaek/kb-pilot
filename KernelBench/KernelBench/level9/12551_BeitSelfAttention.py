from _paritybench_helpers import _mock_config
import math
import torch
from torch import nn
import torch.utils.checkpoint


class BeitRelativePositionBias(nn.Module):

    def __init__(self, config, window_size):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 *
            window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(torch.zeros(self.
            num_relative_distance, config.num_attention_heads))
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:,
            None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(size=(window_size[0] *
            window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1
        self.register_buffer('relative_position_index', relative_position_index
            )

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[self.
            relative_position_index.view(-1)].view(self.window_size[0] *
            self.window_size[1] + 1, self.window_size[0] * self.window_size
            [1] + 1, -1)
        return relative_position_bias.permute(2, 0, 1).contiguous()


class BeitSelfAttention(nn.Module):

    def __init__(self, config, window_size=None):
        super().__init__()
        if (config.hidden_size % config.num_attention_heads != 0 and not
            hasattr(config, 'embedding_size')):
            raise ValueError(
                f'The hidden size {config.hidden_size,} is not a multiple of the number of attention heads {config.num_attention_heads}.'
                )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.
            num_attention_heads)
        self.all_head_size = (self.num_attention_heads * self.
            attention_head_size)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False
            )
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        if window_size:
            self.relative_position_bias = BeitRelativePositionBias(config,
                window_size=window_size)
        else:
            self.relative_position_bias = None

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.
            attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, head_mask=None, output_attentions=
        False, relative_position_bias=None):
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
            -2))
        attention_scores = attention_scores / math.sqrt(self.
            attention_head_size)
        if self.relative_position_bias is not None:
            attention_scores = attention_scores + self.relative_position_bias(
                ).unsqueeze(0)
        if relative_position_bias is not None:
            attention_scores = attention_scores + relative_position_bias
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.
            all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (
            context_layer,)
        return outputs


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4, num_attention_heads=
        4, attention_probs_dropout_prob=0.5)}]
