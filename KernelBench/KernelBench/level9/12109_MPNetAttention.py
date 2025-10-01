from _paritybench_helpers import _mock_config
import math
import torch
from typing import List
from typing import Tuple
from torch import nn
from typing import Set
import torch.utils.checkpoint


def find_pruneable_heads_and_indices(heads: 'List[int]', n_heads: 'int',
    head_size: 'int', already_pruned_heads: 'Set[int]') ->Tuple[Set[int],
    torch.LongTensor]:
    """
    Finds the heads and their indices taking :obj:`already_pruned_heads` into account.

    Args:
        heads (:obj:`List[int]`): List of the indices of heads to prune.
        n_heads (:obj:`int`): The number of heads in the model.
        head_size (:obj:`int`): The size of each head.
        already_pruned_heads (:obj:`Set[int]`): A set of already pruned heads.

    Returns:
        :obj:`Tuple[Set[int], torch.LongTensor]`: A tuple with the remaining heads and their corresponding indices.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: 'torch.LongTensor' = torch.arange(len(mask))[mask].long()
    return heads, index


def prune_linear_layer(layer: 'nn.Linear', index: 'torch.LongTensor', dim:
    'int'=0) ->nn.Linear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (:obj:`torch.nn.Linear`): The layer to prune.
        index (:obj:`torch.LongTensor`): The indices to keep in the layer.
        dim (:obj:`int`, `optional`, defaults to 0): The dimension on which to keep the indices.

    Returns:
        :obj:`torch.nn.Linear`: The pruned layer as a new layer with :obj:`requires_grad=True`.
    """
    index = index
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None
        )
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


class MPNetSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if (config.hidden_size % config.num_attention_heads != 0 and not
            hasattr(config, 'embedding_size')):
            raise ValueError(
                f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})'
                )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.
            num_attention_heads)
        self.all_head_size = (self.num_attention_heads * self.
            attention_head_size)
        self.q = nn.Linear(config.hidden_size, self.all_head_size)
        self.k = nn.Linear(config.hidden_size, self.all_head_size)
        self.v = nn.Linear(config.hidden_size, self.all_head_size)
        self.o = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.
            attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
        position_bias=None, output_attentions=False, **kwargs):
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.
            attention_head_size)
        if position_bias is not None:
            attention_scores += position_bias
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        c = torch.matmul(attention_probs, v)
        c = c.permute(0, 2, 1, 3).contiguous()
        new_c_shape = c.size()[:-2] + (self.all_head_size,)
        c = c.view(*new_c_shape)
        o = self.o(c)
        outputs = (o, attention_probs) if output_attentions else (o,)
        return outputs


class MPNetAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = MPNetSelfAttention(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.
            layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.attn.
            num_attention_heads, self.attn.attention_head_size, self.
            pruned_heads)
        self.attn.q = prune_linear_layer(self.attn.q, index)
        self.attn.k = prune_linear_layer(self.attn.k, index)
        self.attn.v = prune_linear_layer(self.attn.v, index)
        self.attn.o = prune_linear_layer(self.attn.o, index, dim=1)
        self.attn.num_attention_heads = self.attn.num_attention_heads - len(
            heads)
        self.attn.all_head_size = (self.attn.attention_head_size * self.
            attn.num_attention_heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
        position_bias=None, output_attentions=False, **kwargs):
        self_outputs = self.attn(hidden_states, attention_mask, head_mask,
            position_bias, output_attentions=output_attentions)
        attention_output = self.LayerNorm(self.dropout(self_outputs[0]) +
            hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4, num_attention_heads=
        4, attention_probs_dropout_prob=0.5, layer_norm_eps=1,
        hidden_dropout_prob=0.5)}]
