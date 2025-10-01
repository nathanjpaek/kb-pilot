import math
import torch
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import List
from typing import Dict
from typing import Union
from typing import Any
import torch.utils.data
import torch.nn.functional as F
import torch.nn
import torch.cuda
import torch.backends.cudnn
import torch.optim
import torch.cuda.amp


class LayerWithVisualization(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.visualization_enabled = False

    def prepare(self):
        pass

    def plot(self, options: 'Dict[str, Any]') ->Dict[str, Any]:
        raise NotImplementedError()


class AttentionMergeMixin:

    def __init__(self, out_size: 'Optional[int]') ->None:
        self.multi_head_merge = torch.nn.Linear(self.n_heads * self.
            projection_size, out_size or self.state_size, bias=False)

    def merged_attention(self, n_batch: 'int', n_out_steps: 'int', *args,
        need_weights: bool=False, **kwargs) ->Union[torch.Tensor, Tuple[
        torch.Tensor, torch.Tensor]]:
        data, scores = self._attention(*args, **kwargs)
        data = data.view(n_batch, self.n_heads, n_out_steps, -1).permute(0,
            2, 1, 3).contiguous().view(n_batch, n_out_steps, -1)
        return self.multi_head_merge(data), scores


class MultiHeadAttentionBase(LayerWithVisualization):

    def __init__(self, state_size: 'int', n_heads: 'int', dropout: 'float'=
        0.1, projection_size: 'Optional[int]'=None):
        assert state_size % n_heads == 0
        super().__init__()
        self.attention_to_visualize = []
        self.state_size = state_size
        self.projection_size = projection_size or state_size // n_heads
        self.n_heads = n_heads
        self.scale = 1.0 / math.sqrt(self.projection_size)
        self.dropout = torch.nn.Dropout(dropout)

    @staticmethod
    def apply_logit_masks(logits: 'torch.Tensor', mask:
        'Optional[AttentionMask]', val: 'float'=float('-inf')) ->torch.Tensor:
        if mask.position_mask is not None:
            logits = logits.masked_fill(mask.position_mask, val)
        if mask.src_length_mask is not None:
            b, i = mask.src_length_mask.shape
            pad_dims = logits.ndim - 2
            logits = logits.masked_fill(mask.src_length_mask.view([b] + [1] *
                pad_dims + [i]), val)
        return logits

    def _masked_softmax(self, logits: 'torch.Tensor', mask:
        'Optional[AttentionMask]') ->torch.Tensor:
        if (mask is None or mask.src_length_mask is None and mask.
            position_mask is None):
            return F.softmax(logits, -1)
        bb, n_time_dest, n_time_src = logits.shape
        logits = logits.view(bb // self.n_heads, self.n_heads, n_time_dest,
            n_time_src)
        logits = self.apply_logit_masks(logits, mask)
        logits = F.softmax(logits, -1)
        return logits.view(bb, n_time_dest, n_time_src)

    def _attention_read(self, mask: 'Optional[AttentionMask]', scores:
        'torch.Tensor', v: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        s_reshape = scores.view(-1, self.n_heads, *scores.shape[1:])
        if self.visualization_enabled:
            self.attention_to_visualize.append(s_reshape[0])
        return torch.bmm(scores, v), s_reshape

    def transform_data(self, input: 'torch.Tensor', proj:
        'Callable[[torch.Tensor], torch.Tensor]', n_projs: 'int') ->List[torch
        .Tensor]:
        n_batch, n_steps, _ = input.shape
        transformed = proj(input).view(n_batch, n_steps, self.n_heads,
            n_projs, -1).permute(0, 2, 1, 3, 4).contiguous().view(n_batch *
            self.n_heads, n_steps, n_projs, -1)
        return transformed.unbind(dim=2)

    def plot(self, options: 'Dict[str, Any]') ->Dict[str, Any]:
        r = {}
        marks = options.get('steplabel')
        if options.get('mha.plot_head_details'
            ) and self.attention_to_visualize[0].shape[0] > 1:
            for head in range(self.attention_to_visualize[0].shape[0]):
                r[f'head_{head}'] = framework.visualize.plot.AnimatedHeatmap(
                    torch.stack([layer[head] for _, layer in enumerate(self
                    .attention_to_visualize)], 0), ylabel='dest', xlabel=
                    'src', textval=False, x_marks=marks, y_marks=marks,
                    ignore_wrong_marks=True)
        r['attention_max'] = framework.visualize.plot.AnimatedHeatmap(torch
            .stack([layer.max(0)[0] for _, layer in enumerate(self.
            attention_to_visualize)], 0), ylabel='dest', xlabel='src',
            textval=False, x_marks=marks, y_marks=marks, ignore_wrong_marks
            =True)
        self.attention_to_visualize = []
        return r


class AbsPosAttentionBase(MultiHeadAttentionBase):

    def get_attention_scores(self, mask: 'Optional[torch.Tensor]', q:
        'torch.Tensor', k: 'torch.Tensor') ->torch.Tensor:
        logits = torch.bmm(q, k.transpose(1, 2))
        return self._masked_softmax(logits * self.scale, mask)

    def _attention(self, mask: 'Optional[torch.Tensor]', q: 'torch.Tensor',
        k: 'torch.Tensor', v: 'torch.Tensor') ->torch.Tensor:
        scores = self.get_attention_scores(mask, q, k)
        return self._attention_read(mask, scores, v)


class MultiHeadAttention(AttentionMergeMixin, AbsPosAttentionBase):

    def __init__(self, state_size: 'int', n_heads: 'int', dropout: 'float'=
        0.1, input_size: 'Optional[int]'=None, out_size: 'Optional[int]'=None):
        super(AttentionMergeMixin, self).__init__(state_size, n_heads, dropout)
        self.data_to_kv = torch.nn.Linear(state_size, 2 * n_heads * self.
            projection_size, bias=False)
        self.data_to_q = torch.nn.Linear(input_size or state_size, n_heads *
            self.projection_size, bias=False)
        super(MultiHeadAttention, self).__init__(out_size)
        self.reset_parameters()

    def forward(self, curr_state: 'torch.Tensor', attend_to: 'torch.Tensor',
        mask: 'Optional[AttentionMask]', need_weights: 'bool'=False):
        k, v = self.transform_data(attend_to, self.data_to_kv, 2)
        q, = self.transform_data(curr_state, self.data_to_q, 1)
        data, scores = self.merged_attention(curr_state.shape[0], q.shape[1
            ], mask, q, k, v)
        if need_weights:
            return data, scores
        else:
            return data

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.data_to_q.weight)
        torch.nn.init.xavier_uniform_(self.data_to_kv.weight)
        torch.nn.init.xavier_uniform_(self.data_to_kv.weight)


class TransformerEncoderLayer(torch.nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
        activation: 'ActivationFunction'=F.relu, attention_dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=
            attention_dropout)
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.activation = activation
        self.reset_parameters()

    def forward(self, src: 'torch.Tensor', mask: 'Optional[AttentionMask]'=None
        ) ->torch.Tensor:
        src2 = self.self_attn(src, src, mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight, gain=torch.nn.
            init.calculate_gain('relu') if self.activation is F.relu else 1.0)
        torch.nn.init.xavier_uniform_(self.linear2.weight)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'nhead': 4}]
