import torch


class Quantinizer(torch.nn.Module):

    def __init__(self, size):
        super(Quantinizer, self).__init__()
        self.size = size

    def forward(self, x):
        x = (x * self.size * 0.999).long()
        return torch.nn.functional.one_hot(x, num_classes=self.size).float()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size': 4}]
