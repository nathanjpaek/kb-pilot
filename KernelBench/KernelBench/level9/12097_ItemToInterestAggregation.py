import torch
import torch.nn as nn


class ItemToInterestAggregation(nn.Module):

    def __init__(self, seq_len, hidden_size, k_interests=5):
        super().__init__()
        self.k_interests = k_interests
        self.theta = nn.Parameter(torch.randn([hidden_size, k_interests]))

    def forward(self, input_tensor):
        D_matrix = torch.matmul(input_tensor, self.theta)
        D_matrix = nn.Softmax(dim=-2)(D_matrix)
        result = torch.einsum('nij, nik -> nkj', input_tensor, D_matrix)
        return result


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'seq_len': 4, 'hidden_size': 4}]
