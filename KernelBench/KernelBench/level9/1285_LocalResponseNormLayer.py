import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalResponseNormLayer(nn.Module):

    def forward(self, tensor, size=5, alpha=9.999999747378752e-05, beta=
        0.75, k=1.0):
        return F.local_response_norm(tensor, size=size, alpha=alpha, beta=
            beta, k=k)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
