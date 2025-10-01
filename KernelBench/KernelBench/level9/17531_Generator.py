import torch
import torch.onnx
import torch.nn as nn


def outputActivation(x):
    muX = x[:, :, 0:1]
    muY = x[:, :, 1:2]
    sigX = x[:, :, 2:3]
    sigY = x[:, :, 3:4]
    rho = x[:, :, 4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho], dim=2)
    return out


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model, tgt_params):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, tgt_params)

    def forward(self, x):
        fut_pred = self.proj(x)
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = outputActivation(fut_pred)
        return fut_pred


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'tgt_params': 4}]
