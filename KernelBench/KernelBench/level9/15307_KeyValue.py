import torch
import torch.nn
import torch.utils.data.dataset


class KeyValue(torch.nn.Module):

    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.key_conv = torch.nn.Conv2d(indim, keydim, kernel_size=3,
            padding=1, stride=1)
        self.value_conv = torch.nn.Conv2d(indim, valdim, kernel_size=3,
            padding=1, stride=1)

    def forward(self, x):
        return self.key_conv(x), self.value_conv(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'indim': 4, 'keydim': 4, 'valdim': 4}]
