import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Applies an attention mechanism on the query features from the decoder.

    .. math::
            \\begin{array}{ll}
            x = context*query \\\\
            attn_scores = exp(x_i) / sum_j exp(x_j) \\\\
            attn_out = attn * context
            \\end{array}

    Args:
        dim(int): The number of expected features in the query

    Inputs: query, context
        - **query** (batch, query_len, dimensions): tensor containing the query features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: query, attn
        - **query** (batch, query_len, dimensions): tensor containing the attended query features from the decoder.
        - **attn** (batch, query_len, input_len): tensor containing attention weights.

    Attributes:
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.

    """

    def __init__(self):
        super(Attention, self).__init__()
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask
    """
        - query   (batch, query_len, dimensions): tensor containing the query features from the decoder.
        - context (batch, input_len, dimensions): tensor containing features of the encoded input sequence.
    """

    def forward(self, query, context):
        batch_size = query.size(0)
        query.size(2)
        in_len = context.size(1)
        attn = torch.bmm(query, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn_scores = F.softmax(attn.view(-1, in_len), dim=1).view(batch_size,
            -1, in_len)
        attn_out = torch.bmm(attn_scores, context)
        return attn_out, attn_scores


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
