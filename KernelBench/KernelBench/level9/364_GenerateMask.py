import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorAttention(nn.Module):
    """vector attention"""

    def __init__(self, input_dim, hidden_dim):
        super(VectorAttention, self).__init__()
        self.theta = nn.Linear(input_dim, hidden_dim)
        self.phi = nn.Linear(input_dim, hidden_dim)
        self.psi = nn.Linear(input_dim, hidden_dim)
        self.recover = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x_t = self.theta(x)
        x_ph = self.phi(x)
        x_psi = self.psi(x)
        attention_map = torch.matmul(x_ph, torch.transpose(x_t, 0, 1))
        attention_map = attention_map
        attention_map = F.softmax(attention_map, dim=1)
        x_add = torch.matmul(attention_map, x_psi)
        x_add = self.recover(x_add)
        return x + x_add


class GenerateMask(nn.Module):
    """
    sparsify by 1x1 conv and apply gumbel softmax
    """

    def __init__(self, ch, resolution, or_cadidate=1000, no_attention=False):
        super(GenerateMask, self).__init__()
        self.conv = nn.Conv2d(ch, or_cadidate, kernel_size=1, padding=0,
            bias=False)
        self.relu = nn.ReLU()
        self.no_attention = no_attention
        if not self.no_attention:
            self.prob_attention = VectorAttention(input_dim=or_cadidate,
                hidden_dim=or_cadidate // 3)

    def forward(self, x):
        """
        input x: [n, L, c, kernel, kernel]
        output mask: [n * L, 1, h, w]
        """
        n, L, c, h, w = x.shape
        y = self.conv(x.reshape(-1, c, h, w))
        prob_vector = F.softmax(y.sum((2, 3)), dim=1)
        prob_vector_previous = prob_vector
        if not self.no_attention:
            prob_vector = self.prob_attention(prob_vector)
        prob_vector = F.softmax(prob_vector)
        weighted_mask = (prob_vector[:, :, None, None] * y).sum((1,),
            keepdim=True)
        weighted_mask = F.softmax(weighted_mask.view(n * L, 1, h * w), dim=2
            ).view(n * L, 1, h, w)
        return weighted_mask, prob_vector, prob_vector_previous, y


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'ch': 4, 'resolution': 4}]
