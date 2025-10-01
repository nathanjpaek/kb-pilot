import torch


class PointwiseConvolutionLayer(torch.nn.Module):

    def __init__(self, N, F, F_prime):
        super().__init__()
        self.f1 = torch.nn.Linear(F, 128)
        self.f2 = torch.nn.Linear(128, F_prime)

    def forward(self, f_bar_batch):
        output = torch.nn.functional.softplus(self.f1(f_bar_batch))
        return torch.nn.functional.softplus(self.f2(output))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'N': 4, 'F': 4, 'F_prime': 4}]
