import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, ft_dim, rnn_dim, attn_dim):
        super().__init__()
        self.enc_attn = nn.Linear(ft_dim, attn_dim)
        self.dec_attn = nn.Linear(rnn_dim, attn_dim)
        self.attn = nn.Linear(attn_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    """
    Args:
        torch.Tensor feature:               (bs x * x ft_dim)
        torch.Tensor memory:                (bs x rnn_dim)
    Returns:
        torch.Tensor attn_weights:          (bs x *)
        torch.Tensor weighted_feature:      (bs x ft_dim)
    """

    def forward(self, feature, memory):
        encoded_feature = self.enc_attn(feature)
        encoded_memory = self.dec_attn(memory).unsqueeze(1)
        attn_weights = self.attn(self.relu(encoded_feature + encoded_memory)
            ).squeeze(-1)
        attn_weights = self.softmax(attn_weights)
        weighted_feature = (feature * attn_weights.unsqueeze(-1)).sum(dim=1)
        return attn_weights, weighted_feature


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'ft_dim': 4, 'rnn_dim': 4, 'attn_dim': 4}]
