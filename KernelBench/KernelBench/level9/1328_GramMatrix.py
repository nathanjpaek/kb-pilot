import torch
import torch.nn as nn
import torch.onnx


class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()
        features = input.view(a, b, c * d)
        G = torch.matmul(features, torch.transpose(features, 1, 2))
        return G.div(a * b * c * d)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
