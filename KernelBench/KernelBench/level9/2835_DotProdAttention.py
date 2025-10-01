import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProdAttention(nn.Module):
    """Basic Dot-Production Attention"""

    def __init__(self):
        super().__init__()

    def forward(self, output, context):
        """Basic Dot-Production Method
            1. compute e = q * k
            2. compute tanh(softmax(e) * k)

            Args:
                output (batch, 1, hidden): output from decoder rnn
                context (batch, seq, hidden): output from encoder rnn
            Returns:
                output (batch, 1, hidden): modified output
                attn (batch, 1, seq): attention state in this step
        """
        attn = torch.bmm(output, context.transpose(1, 2))
        attn = F.softmax(attn, dim=2)
        output = F.tanh(torch.bmm(attn, context))
        return output, attn


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
