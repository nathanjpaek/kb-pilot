import torch
from torch import nn
import torch.nn.functional as F
from scipy.cluster.vq import kmeans2


class VQVAEQuantize(nn.Module):
    """
    Neural Discrete Representation Learning, van den Oord et al. 2017
    https://arxiv.org/abs/1711.00937

    Follows the original DeepMind implementation
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
    """

    def __init__(self, num_hiddens, n_embed, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_embed = n_embed
        self.kld_scale = 10.0
        self.proj = nn.Conv2d(num_hiddens, embedding_dim, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)
        self.register_buffer('data_initialized', torch.zeros(1))

    def forward(self, z):
        B, _C, H, W = z.size()
        z_e = self.proj(z)
        z_e = z_e.permute(0, 2, 3, 1)
        flatten = z_e.reshape(-1, self.embedding_dim)
        if self.training and self.data_initialized.item() == 0:
            None
            rp = torch.randperm(flatten.size(0))
            kd = kmeans2(flatten[rp[:20000]].data.cpu().numpy(), self.
                n_embed, minit='points')
            self.embed.weight.data.copy_(torch.from_numpy(kd[0]))
            self.data_initialized.fill_(1)
        dist = flatten.pow(2).sum(1, keepdim=True
            ) - 2 * flatten @ self.embed.weight.t() + self.embed.weight.pow(2
            ).sum(1, keepdim=True).t()
        _, ind = (-dist).max(1)
        ind = ind.view(B, H, W)
        z_q = self.embed_code(ind)
        commitment_cost = 0.25
        diff = commitment_cost * (z_q.detach() - z_e).pow(2).mean() + (z_q -
            z_e.detach()).pow(2).mean()
        diff *= self.kld_scale
        z_q = z_e + (z_q - z_e).detach()
        z_q = z_q.permute(0, 3, 1, 2)
        return z_q, diff, ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.weight)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_hiddens': 4, 'n_embed': 4, 'embedding_dim': 4}]
