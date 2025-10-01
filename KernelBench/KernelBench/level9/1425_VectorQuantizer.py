from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F


class VQPseudoGrad(Function):

    @staticmethod
    def forward(ctx, z, q):
        return q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class VectorQuantizer(nn.Module):

    def __init__(self, k: 'int', d: 'int', commitment_cost: 'float'=0.25):
        super().__init__()
        self.commitment_cost = commitment_cost
        self.embedding = nn.Parameter(torch.empty(k, d))
        nn.init.uniform_(self.embedding, -1 / k, 1 / k)

    def forward(self, z):
        _b, _c, _h, _w = z.size()
        z_ = z.permute(0, 2, 3, 1)
        e = self.embedding
        distances = (z_ * z_).sum(-1, keepdim=True) - 2 * torch.einsum(
            '...d, nd -> ...n', z_, e) + (e * e).sum(-1, keepdim=True).t()
        code = distances.argmin(-1)
        zq = F.embedding(code, e).permute(0, 3, 1, 2).contiguous()
        e_latent_loss = F.mse_loss(zq.detach(), z)
        q_latent_loss = F.mse_loss(zq, z.detach())
        loss = q_latent_loss + e_latent_loss * self.commitment_cost
        return VQPseudoGrad.apply(z, zq), loss, code

    def extra_repr(self):
        return (
            f'(embedding): k={self.embedding.size(0)}, d={self.embedding.size(1)}'
            )


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'k': 4, 'd': 4}]
