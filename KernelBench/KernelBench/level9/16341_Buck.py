import torch
import torch.nn


class Buck(torch.nn.Module):

    def __init__(self, A=1.0, B=1.0, C=1.0):
        super(Buck, self).__init__()
        self.A = torch.nn.Parameter(torch.Tensor([A]))
        self.B = torch.nn.Parameter(torch.Tensor([B]))
        self.C = torch.nn.Parameter(torch.Tensor([C]))

    def Buckingham(self, r, A, B, C):
        return A * torch.exp(-B * r) - C / r ** 6

    def forward(self, x):
        return self.Buckingham(x, self.A, self.B, self.C)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
