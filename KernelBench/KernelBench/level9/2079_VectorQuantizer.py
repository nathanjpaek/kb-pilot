import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(self, num_embeddings: 'int', embedding_dim: 'int', beta:
        'float'=0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: 'Tensor') ->Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + torch.sum(
            self.embedding.weight ** 2, dim=1) - 2 * torch.matmul(flat_latents,
            self.embedding.weight.t())
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K,
            device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.
            weight)
        quantized_latents = quantized_latents.view(latents_shape)
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())
        vq_loss = commitment_loss * self.beta + embedding_loss
        quantized_latents = latents + (quantized_latents - latents).detach()
        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_embeddings': 4, 'embedding_dim': 4}]
