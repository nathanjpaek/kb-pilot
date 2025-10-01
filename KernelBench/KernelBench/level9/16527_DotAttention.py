import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class DotAttention(nn.Module):

    def __init__(self, hidden_size):
        super(DotAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attn_vector = nn.Parameter(torch.Tensor(1, hidden_size),
            requires_grad=True)
        init.xavier_uniform(self.attn_vector.data)

    def get_mask(self):
        pass

    def forward(self, inputs, lengths=None):
        batch_size, _max_len = inputs.size()[:2]
        """
        print("INPUTS", inputs.size())
        print("ATTN", self.attn_vector  # (1, hidden_size)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .transpose(2, 1)
                            .repeat(batch_size, 1, 1).size())"""
        weights = torch.bmm(inputs, self.attn_vector.unsqueeze(0).transpose
            (2, 1).repeat(batch_size, 1, 1))
        attn_energies = F.softmax(F.relu(weights.squeeze()))
        _sums = attn_energies.sum(-1).unsqueeze(1).expand_as(attn_energies)
        attn_weights = attn_energies / _sums
        weighted = torch.mul(inputs, attn_weights.unsqueeze(-1).expand_as(
            inputs))
        representations = weighted.sum(1).squeeze()
        return representations, attn_weights


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
