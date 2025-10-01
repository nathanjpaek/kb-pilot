import torch
import torch.nn as nn
import torch.onnx


class Norm(nn.Module):

    def __init__(self, emb_dim, eps=1e-06):
        super().__init__()
        self.size = emb_dim
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        """
        inputs:
            x: input of shape: (batch size, sequence length, embedding dimensions)

        outputs: Scaled, normalized x
        """
        norm = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=
            True) + self.eps)
        norm = self.alpha * norm + self.bias
        return norm


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'emb_dim': 4}]
