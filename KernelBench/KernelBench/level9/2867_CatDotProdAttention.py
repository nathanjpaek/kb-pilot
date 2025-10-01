import torch
import torch.nn as nn
import torch.nn.functional as F


class CatDotProdAttention(nn.Module):
    """Dot-Production Attention concatenated with query values
        Attribute:
            linear (nn.Linear): linear layer to compress output
    """

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim * 2, dim)

    def forward(self, output, context):
        """Variation of Dot-Production Method
            1. compute e = q * k
            2. compute prod = softmax(e) * k
            3. concatenate prod with q as output
            4. compute and return tanh(linear_layer(output))

            Args:
                output (batch, 1, hidden): output from decoder rnn
                context (batch, seq, hidden): output from encoder rnn
            Returns:
                output (batch, 1, hidden): modified output
                attn (batch, 1, seq): attention state in this step
        """
        attn = torch.bmm(output, context.transpose(1, 2))
        attn = F.softmax(attn, dim=2)
        prod = torch.bmm(attn, context)
        output = torch.cat([prod, output], dim=2)
        output = F.tanh(self.linear(output))
        return output, attn


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
