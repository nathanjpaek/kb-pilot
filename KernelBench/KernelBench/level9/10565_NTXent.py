import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class NTXent(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    code: https://github.com/AndrewAtanov/simclr-pytorch/blob/master/models/losses.py
    """
    LARGE_NUMBER = 1000000000.0

    def __init__(self, tau=1.0, gpu=None, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.norm = 1.0

    def forward(self, batch_sample_one, batch_sample_two):
        z = torch.cat([batch_sample_one, batch_sample_two], dim=0)
        n = z.shape[0]
        assert n % self.multiplier == 0
        z = F.normalize(z, p=2, dim=1) / np.sqrt(self.tau)
        logits = z @ z.t()
        logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER
        logprob = F.log_softmax(logits, dim=1)
        m = self.multiplier
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n //
            m, n)) % n
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)
        loss = -logprob[np.repeat(np.arange(n), m - 1), labels].sum() / n / (m
             - 1) / self.norm
        return loss


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
