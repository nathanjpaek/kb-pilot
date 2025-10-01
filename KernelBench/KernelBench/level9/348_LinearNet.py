import torch
import torch.nn
import torch.optim


class LinearNet(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.nonlinear = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x: 'torch.Tensor'):
        x = x.requires_grad_(True)
        x = torch.nn.functional.normalize(x)
        x = self.linear1(x)
        x = self.nonlinear(x)
        x = self.linear2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'D_in': 4, 'H': 4, 'D_out': 4}]
