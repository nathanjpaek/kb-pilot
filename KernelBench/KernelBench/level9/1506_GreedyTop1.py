import torch
from typing import Optional
import torch as pt
import torch.distributed
import torch.distributed.elastic.multiprocessing.errors


class GreedyTop1(pt.nn.Module):
    """
    Implements picking the highest scoring next word with support for vocabulary selection and target factors.
    """

    def forward(self, scores: 'pt.Tensor', vocab_slice_ids:
        'Optional[pt.Tensor]'=None, target_factors: 'Optional[pt.Tensor]'=None
        ) ->pt.Tensor:
        best_word_index = pt.argmin(scores, dim=-1, keepdim=True)
        if vocab_slice_ids is not None:
            best_word_index = vocab_slice_ids.index_select(0,
                best_word_index.squeeze(1)).unsqueeze(1)
        if target_factors is not None:
            factor_index = target_factors[:, :, 1].int()
            best_word_index = pt.cat((best_word_index, factor_index), dim=1)
        return best_word_index


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
