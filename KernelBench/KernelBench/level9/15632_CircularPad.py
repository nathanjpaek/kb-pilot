import torch


class CircularPad(torch.nn.Module):

    def __init__(self, padding=(1, 1, 0, 0)):
        super(CircularPad, self).__init__()
        self.padding = padding

    def forward(self, input):
        return torch.nn.functional.pad(input=input, pad=self.padding, mode=
            'circular')


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
