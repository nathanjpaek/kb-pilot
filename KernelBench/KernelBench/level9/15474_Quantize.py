import torch
from torch import nn
from torch.nn import functional as F


class Quantize(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, dim, n_embed, beta=0.25):
        super().__init__()
        self.n_e = n_embed
        self.e_dim = dim
        self.beta = beta
        rand_range = 1.0 / self.n_e
        self.embeddings = nn.Parameter(torch.rand(dim, n_embed).mul_(2 *
            rand_range).sub_(rand_range))

    def forward(self, input):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        x = input.permute(0, 2, 3, 1)
        flatten = x.reshape(-1, x.size(-1))
        dist = flatten.pow(2).sum(1, keepdim=True
            ) - 2 * flatten @ self.embeddings + self.embeddings.pow(2).sum(
            0, keepdim=True)
        _, embed_ind = (-dist).max(1)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind)
        loss = torch.mean((quantize.detach() - input).pow(2)
            ) + self.beta * torch.mean((quantize - input.detach()).pow(2))
        quantize = input + (quantize - input).detach()
        return quantize, loss, embed_ind

    def embed_code(self, embed_id):
        codes = F.embedding(embed_id, self.embeddings.transpose(0, 1))
        return codes.permute(0, 3, 1, 2).contiguous()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'n_embed': 4}]
