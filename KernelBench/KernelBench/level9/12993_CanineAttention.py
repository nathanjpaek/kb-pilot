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


class CanineSelfAttention(nn.Module):

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
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
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

    def forward(self, from_tensor, to_tensor, attention_mask=None,
        head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(from_tensor)
        key_layer = self.transpose_for_scores(self.key(to_tensor))
        value_layer = self.transpose_for_scores(self.value(to_tensor))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
            -2))
        if (self.position_embedding_type == 'relative_key' or self.
            position_embedding_type == 'relative_key_query'):
            seq_length = from_tensor.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long,
                device=from_tensor.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long,
                device=from_tensor.device).view(1, -1)
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
        attention_scores = attention_scores / math.sqrt(self.
            attention_head_size)
        if attention_mask is not None:
            if attention_mask.ndim == 3:
                attention_mask = torch.unsqueeze(attention_mask, dim=1)
                attention_mask = (1.0 - attention_mask.float()) * -10000.0
            attention_scores = attention_scores + attention_mask
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


class CanineSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.
            layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CanineAttention(nn.Module):
    """
    Additional arguments related to local attention:

        - **local** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Whether to apply local attention.
        - **always_attend_to_first_position** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Should all blocks
          be able to attend
        to the :obj:`to_tensor`'s first position (e.g. a [CLS] position)? - **first_position_attends_to_all**
        (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Should the `from_tensor`'s first position be able to
        attend to all positions within the `from_tensor`? - **attend_from_chunk_width** (:obj:`int`, `optional`,
        defaults to 128) -- The width of each block-wise chunk in :obj:`from_tensor`. - **attend_from_chunk_stride**
        (:obj:`int`, `optional`, defaults to 128) -- The number of elements to skip when moving to the next block in
        :obj:`from_tensor`. - **attend_to_chunk_width** (:obj:`int`, `optional`, defaults to 128) -- The width of each
        block-wise chunk in `to_tensor`. - **attend_to_chunk_stride** (:obj:`int`, `optional`, defaults to 128) -- The
        number of elements to skip when moving to the next block in :obj:`to_tensor`.
    """

    def __init__(self, config, local=False, always_attend_to_first_position:
        'bool'=False, first_position_attends_to_all: 'bool'=False,
        attend_from_chunk_width: 'int'=128, attend_from_chunk_stride: 'int'
        =128, attend_to_chunk_width: 'int'=128, attend_to_chunk_stride:
        'int'=128):
        super().__init__()
        self.self = CanineSelfAttention(config)
        self.output = CanineSelfOutput(config)
        self.pruned_heads = set()
        self.local = local
        if attend_from_chunk_width < attend_from_chunk_stride:
            raise ValueError(
                '`attend_from_chunk_width` < `attend_from_chunk_stride`would cause sequence positions to get skipped.'
                )
        if attend_to_chunk_width < attend_to_chunk_stride:
            raise ValueError(
                '`attend_to_chunk_width` < `attend_to_chunk_stride`would cause sequence positions to get skipped.'
                )
        self.always_attend_to_first_position = always_attend_to_first_position
        self.first_position_attends_to_all = first_position_attends_to_all
        self.attend_from_chunk_width = attend_from_chunk_width
        self.attend_from_chunk_stride = attend_from_chunk_stride
        self.attend_to_chunk_width = attend_to_chunk_width
        self.attend_to_chunk_stride = attend_to_chunk_stride

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.self.
            num_attention_heads, self.self.attention_head_size, self.
            pruned_heads)
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(
            heads)
        self.self.all_head_size = (self.self.attention_head_size * self.
            self.num_attention_heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
        output_attentions=False):
        if not self.local:
            self_outputs = self.self(hidden_states, hidden_states,
                attention_mask, head_mask, output_attentions)
            attention_output = self_outputs[0]
        else:
            from_seq_length = to_seq_length = hidden_states.shape[1]
            from_tensor = to_tensor = hidden_states
            from_chunks = []
            if self.first_position_attends_to_all:
                from_chunks.append((0, 1))
                from_start = 1
            else:
                from_start = 0
            for chunk_start in range(from_start, from_seq_length, self.
                attend_from_chunk_stride):
                chunk_end = min(from_seq_length, chunk_start + self.
                    attend_from_chunk_width)
                from_chunks.append((chunk_start, chunk_end))
            to_chunks = []
            if self.first_position_attends_to_all:
                to_chunks.append((0, to_seq_length))
            for chunk_start in range(0, to_seq_length, self.
                attend_to_chunk_stride):
                chunk_end = min(to_seq_length, chunk_start + self.
                    attend_to_chunk_width)
                to_chunks.append((chunk_start, chunk_end))
            if len(from_chunks) != len(to_chunks):
                raise ValueError(
                    f'Expected to have same number of `from_chunks` ({from_chunks}) and `to_chunks` ({from_chunks}). Check strides.'
                    )
            attention_output_chunks = []
            attention_probs_chunks = []
            for (from_start, from_end), (to_start, to_end) in zip(from_chunks,
                to_chunks):
                from_tensor_chunk = from_tensor[:, from_start:from_end, :]
                to_tensor_chunk = to_tensor[:, to_start:to_end, :]
                attention_mask_chunk = attention_mask[:, from_start:
                    from_end, to_start:to_end]
                if self.always_attend_to_first_position:
                    cls_attention_mask = attention_mask[:, from_start:
                        from_end, 0:1]
                    attention_mask_chunk = torch.cat([cls_attention_mask,
                        attention_mask_chunk], dim=2)
                    cls_position = to_tensor[:, 0:1, :]
                    to_tensor_chunk = torch.cat([cls_position,
                        to_tensor_chunk], dim=1)
                attention_outputs_chunk = self.self(from_tensor_chunk,
                    to_tensor_chunk, attention_mask_chunk, head_mask,
                    output_attentions)
                attention_output_chunks.append(attention_outputs_chunk[0])
                if output_attentions:
                    attention_probs_chunks.append(attention_outputs_chunk[1])
            attention_output = torch.cat(attention_output_chunks, dim=1)
        attention_output = self.output(attention_output, hidden_states)
        outputs = attention_output,
        if not self.local:
            outputs = outputs + self_outputs[1:]
        else:
            outputs = outputs + tuple(attention_probs_chunks)
        return outputs


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4, num_attention_heads=
        4, attention_probs_dropout_prob=0.5, position_embedding_type=4,
        layer_norm_eps=1, hidden_dropout_prob=0.5)}]
