import torch
from torch import Tensor
import torch.onnx.operators


def create_max_segment_mask(tensor: 'Tensor', max_segment_length):
    """
    Create max-segment mask.

    Args:
        tensor:
            :math: (N, T, *) where T is target dimension

    Returns:
        - max-segment mask:
            :math:`(N, T)` where T is target dimension
    """
    sz = tensor.size(1)
    mask = [[(i <= j < i + max_segment_length) for j in range(sz)] for i in
        range(sz)]
    mask = torch.BoolTensor(mask).type_as(tensor).bool()
    return mask


def create_upper_triangular_mask(tensor: 'Tensor'):
    """
    Create upper triangular mask. It is usually used in auto-regressive model in training

    Args:
        tensor:
            :math: (N, T, *) where T is target dimension

    Returns:
        - upper triangular mask:
            :math:`(N, T)` where T is target dimension
    """
    sz = tensor.size(1)
    mask = (torch.triu(torch.ones(sz, sz)) == 1).type_as(tensor).bool()
    return mask.detach()


class _Extraction(torch.nn.Module):
    """
    Extraction methods transform a pair of start and end position to a segment of context.

    Args:
        pad: pad index
        max_segment_length: maximum length for extracted results
    """

    def __init__(self, pad, max_segment_length=None):
        super().__init__()
        self._pad = pad
        self._max_segment_length = max_segment_length

    def forward(self, context, start_logits, end_logits):
        """
        Extract a piece of content from context

        Args:
            context: whole context for extraction
            start_logits: log probability of start position
            end_logits: log probability of end position

        Returns:
            - an extracted sequence of maximum probability
        """
        attention_mask = context.ne(self._pad)
        start_logits = start_logits.masked_fill(~attention_mask, float('-inf'))
        end_logits = end_logits.masked_fill(~attention_mask, float('-inf'))
        batch_size, seqlen = context.size()
        logits = start_logits.unsqueeze(dim=2) + end_logits.unsqueeze(dim=1)
        mask = create_upper_triangular_mask(context)
        if self._max_segment_length:
            max_segment_mask = create_max_segment_mask(context, self.
                _max_segment_length)
            mask = mask & max_segment_mask
        logits = logits.masked_fill(~mask, float('-inf'))
        logits = logits.view(batch_size, seqlen * seqlen)
        _, pos = logits.max(dim=-1)
        start_pos, end_pos = pos // seqlen, pos % seqlen
        return start_pos, end_pos


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'pad': 4}]
