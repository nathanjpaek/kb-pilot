import torch


class SplitChannels(torch.nn.Module):

    def __init__(self, split_location):
        super(SplitChannels, self).__init__()
        self.split_location = split_location

    def forward(self, x):
        a, b = x[:, :self.split_location], x[:, self.split_location:]
        a, b = a.clone(), b.clone()
        del x
        return a, b

    def inverse(self, x, y):
        return torch.cat([x, y], dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'split_location': 4}]
