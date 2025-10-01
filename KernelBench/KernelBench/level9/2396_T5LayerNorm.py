import torch
import torch.nn as nn
import torch.utils.checkpoint


class T5LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-06):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.
            variance_epsilon)
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states
        return self.weight * hidden_states


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
