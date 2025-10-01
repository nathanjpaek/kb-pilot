import torch


class Rescale(torch.nn.Module):

    def __init__(self, old_min, old_max, new_min, new_max):
        super(Rescale, self).__init__()
        self.old_min = old_min
        self.old_max = old_max
        self.new_min = new_min
        self.new_max = new_max

    def forward(self, x):
        x = (x - self.old_min) / (self.old_max - self.old_min) * (self.
            new_max - self.new_min) + self.new_min
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'old_min': 4, 'old_max': 4, 'new_min': 4, 'new_max': 4}]
