import torch
from torch import nn


class AFTSimple(nn.Module):

    def __init__(self, max_seqlen, dim, hidden_dim=64):
        super().__init__()
        """
        max_seqlen: the maximum number of timesteps (sequence length) to be fed in
        dim: the embedding dimension of the tokens
        hidden_dim: the hidden dimension used inside AFT Full
        
        Number of Heads is 1 as done in the paper.
        """
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.to_q = nn.Linear(dim, hidden_dim)
        self.to_k = nn.Linear(dim, hidden_dim)
        self.to_v = nn.Linear(dim, hidden_dim)
        self.project = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.to_q(x).view(B, T, self.hidden_dim)
        K = self.to_k(x).view(B, T, self.hidden_dim)
        V = self.to_v(x).view(B, T, self.hidden_dim)
        """
        From the paper
        """
        weights = torch.mul(torch.softmax(K, -1), V)
        Q_sig = torch.sigmoid(Q)
        Yt = torch.mul(Q_sig, weights)
        Yt = Yt.view(B, T, self.hidden_dim)
        Yt = self.project(Yt)
        return Yt


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'max_seqlen': 4, 'dim': 4}]
