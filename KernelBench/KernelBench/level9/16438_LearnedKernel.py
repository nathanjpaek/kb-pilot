from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class LearnedKernel(nn.Module):

    def __init__(self, args: 'Namespace'):
        super(LearnedKernel, self).__init__()
        self.A = nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size)

    def forward(self, encodings: 'torch.Tensor'):
        return (self.A(encodings[:, 1, :].squeeze(1)) * encodings[:, 0, :].
            squeeze(1)).sum(dim=1, keepdim=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'args': _mock_config(ffn_hidden_size=4)}]
