import torch


class ExampleTorchModule(torch.nn.Module):

    def __init__(self):
        super(ExampleTorchModule, self).__init__()

    def forward(self, input):
        residual = 10 - input
        return residual


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
