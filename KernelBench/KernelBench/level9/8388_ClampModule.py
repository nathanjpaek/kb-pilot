import torch
import torch as th


class ClampModule(th.nn.Module):
    """Why is this not a thing in the main library?"""

    def __init__(self, min_v, max_v):
        super().__init__()
        self.min_v = min_v
        self.max_v = max_v

    def forward(self, x):
        return th.clamp(x, self.min_v, self.max_v)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'min_v': 4, 'max_v': 4}]
