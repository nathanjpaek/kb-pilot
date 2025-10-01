import torch


class RemoveChannelMeanStd(torch.nn.Module):

    def forward(self, x):
        x2 = x.view(x.size(0), x.size(1), -1)
        mean = x2.mean(dim=2).view(x.size(0), x.size(1), 1, 1)
        std = x2.std(dim=2).view(x.size(0), x.size(1), 1, 1)
        return (x - mean) / std


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
