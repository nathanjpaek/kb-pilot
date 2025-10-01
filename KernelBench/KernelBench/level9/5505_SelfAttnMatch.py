import math
import torch
import torch.nn.functional as F
import torch.nn as nn


class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 
            0.044715 * torch.pow(x, 3))))


class SelfAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * x_j) for i in X
    * alpha_j = softmax(x_j * x_i)
    """

    def __init__(self, input_size, identity=False, diag=True):
        super(SelfAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None
        self.diag = diag
        self.gelu = GELU()

    def forward(self, inputs):
        """
        Args:
            x: batch * len1 * dim1
            x_mask: batch * len1 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * dim1
        """
        if self.linear:
            x_proj = self.linear(inputs)
            x_proj = self.gelu(x_proj)
        else:
            x_proj = inputs
        scores = x_proj.bmm(x_proj.transpose(2, 1))
        if not self.diag:
            x_len = inputs.size(1)
            for i in range(x_len):
                scores[:, i, i] = 0
        alpha = F.softmax(scores, dim=2)
        matched_seq = alpha.bmm(inputs)
        return matched_seq, alpha.sum(dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
