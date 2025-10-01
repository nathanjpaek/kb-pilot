import torch


class DummyLayer(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self, x):
        return x + self.dummy - self.dummy


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
