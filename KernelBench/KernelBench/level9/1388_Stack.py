import torch
import torch.utils
import torch.utils.data


class Stack(torch.nn.Module):

    def __init__(self, repeats):
        super(Stack, self).__init__()
        self.repeats = repeats

    def forward(self, x):
        x = torch.repeat_interleave(x, self.repeats, dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'repeats': 4}]
