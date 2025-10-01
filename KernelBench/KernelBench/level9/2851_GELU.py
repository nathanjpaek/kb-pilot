import math
import torch
import torch.nn as nn
import torch.cuda
import torch.distributed


class GELU(nn.Module):
    """ Implementation of the gelu activation function
        :cite:`DBLP:journals/corr/HendrycksG16`

        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi)
                    * (x + 0.044715 * torch.pow(x, 3))))

        Examples::
        >>> m = GELU()
        >>> inputs = torch.randn(2)
        >>> outputs = m(inputs)
    """

    def forward(self, x):
        gelu = x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        return gelu


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
