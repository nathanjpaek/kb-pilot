import torch
import torch.nn as nn


class kAttentionPooling(nn.Module):

    def __init__(self, seq_len, hidden_size, k_heads=5):
        super().__init__()
        self.k_heads = k_heads
        self.theta_k = nn.Parameter(torch.randn([hidden_size, k_heads]))

    def forward(self, input_tensor):
        attention_matrix = torch.matmul(input_tensor, self.theta_k)
        attention_matrix = nn.Softmax(dim=-2)(attention_matrix)
        pooling_result = torch.einsum('nij, nik -> nkj', input_tensor,
            attention_matrix)
        return pooling_result


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'seq_len': 4, 'hidden_size': 4}]
