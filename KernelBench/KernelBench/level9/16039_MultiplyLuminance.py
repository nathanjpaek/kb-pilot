import torch


class MultiplyLuminance(torch.nn.Module):

    def __init__(self):
        super(MultiplyLuminance, self).__init__()

    def forward(self, color, luminance):
        return color * (1 + luminance)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
