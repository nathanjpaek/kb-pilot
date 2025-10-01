import torch


class PolarityInversion(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, audio):
        audio = torch.neg(audio)
        return audio


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
