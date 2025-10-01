import torch
import torch.nn as nn
import torch.nn.functional as F


class Quantize(nn.Module):

    def __init__(self, emb_dim, emb_size, decay=0.99, eps=1e-05, ema_flag=
        False, bdt_flag=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.emb_size = emb_size
        self.ema_flag = ema_flag
        self.bdt_flag = bdt_flag
        self.embedding = nn.Embedding(emb_size, emb_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.emb_size, 1.0 /
            self.emb_size)
        if self.ema_flag:
            self.decay = decay
            self.eps = eps
            embed = torch.randn(emb_dim, emb_size)
            self.register_buffer('ema_size', torch.zeros(emb_size))
            self.register_buffer('ema_w', embed.clone())

    def forward(self, x, use_ema=True):
        if self.bdt_flag:
            x = x.transpose(1, 2)
        quantized_idx, quantized_onehot = self.vq(x)
        embed_idx = torch.matmul(quantized_onehot.float(), self.embedding.
            weight)
        if self.training and self.ema_flag and use_ema:
            self.ema_size = self.decay * self.ema_size + (1 - self.decay
                ) * torch.sum(quantized_onehot.view(-1, self.emb_size), 0)
            embed_sum = torch.sum(torch.matmul(x.transpose(1, 2),
                quantized_onehot.float()), dim=0)
            self.ema_w.data = self.decay * self.ema_w.data + (1 - self.decay
                ) * embed_sum
            n = torch.sum(self.ema_size)
            self.ema_size = (self.ema_size + self.eps) / (n + self.emb_size *
                self.eps) * n
            embed_normalized = self.ema_w / self.ema_size.unsqueeze(0)
            self.embedding.weight.data.copy_(embed_normalized.transpose(0, 1))
        embed_idx_qx = x + (embed_idx - x).detach()
        if self.bdt_flag:
            embed_idx_qx = embed_idx_qx.transpose(1, 2)
        return embed_idx, embed_idx_qx, quantized_idx

    def vq(self, x):
        flatten_x = x.reshape(-1, self.emb_dim)
        dist = torch.sum(torch.pow(self.embedding.weight, 2), dim=1
            ) - 2 * torch.matmul(flatten_x, self.embedding.weight.T
            ) + torch.sum(torch.pow(flatten_x, 2), dim=1, keepdim=True)
        quantized_idx = torch.argmin(dist, dim=1).view(x.size(0), x.size(1))
        quantized_onehot = F.one_hot(quantized_idx, self.emb_size)
        return quantized_idx, quantized_onehot


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'emb_dim': 4, 'emb_size': 4}]
