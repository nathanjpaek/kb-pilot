import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXent(nn.Module):

    def forward(self, z1, z2, t):
        batch_size = z1.shape[0]
        device = z1.device
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        similarity = torch.matmul(z1, z2.T)
        similarity = similarity * torch.exp(t)
        targets = torch.arange(batch_size, device=device)
        loss = F.cross_entropy(similarity, targets)
        return loss


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
