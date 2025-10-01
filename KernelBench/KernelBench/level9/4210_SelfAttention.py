import torch
import torch.nn as nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    """
	Implementation of the attention block
	"""

    def __init__(self, input_size, hidden_size, output_size):
        super(SelfAttention, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size, bias=False)
        self.layer2 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, attention_input):
        out = self.layer1(attention_input)
        out = torch.tanh(out)
        out = self.layer2(out)
        out = out.permute(0, 2, 1)
        out = F.softmax(out, dim=2)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}]
