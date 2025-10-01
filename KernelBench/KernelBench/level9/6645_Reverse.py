import torch


class Reverse(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, audio):
        return torch.flip(audio, dims=[1])


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
