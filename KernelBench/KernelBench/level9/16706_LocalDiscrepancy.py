import torch
import torch.nn as nn
import torch.backends.cudnn
import torch.utils
import torch.distributed


class LocalDiscrepancy(nn.Module):

    def __init__(self, in_channels=19, padding_mode='replicate', neighbor=8,
        l_type='l1'):
        """
        depth-wise conv to calculate the mean of neighbor
        """
        super(LocalDiscrepancy, self).__init__()
        self.type = l_type
        self.mean_conv = nn.Conv2d(in_channels=in_channels, out_channels=
            in_channels, kernel_size=3, stride=1, padding=int(3 / 2), bias=
            False, padding_mode=padding_mode, groups=in_channels)
        if neighbor == 8:
            a = torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0,
                1.0]]]]) / 9
        elif neighbor == 4:
            a = torch.tensor([[[[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0,
                0.0]]]]) / 5
        else:
            raise NotImplementedError
        a = a.repeat([in_channels, 1, 1, 1])
        a = nn.Parameter(a)
        self.mean_conv.weight = a
        self.mean_conv.requires_grad_(False)

    def forward(self, x):
        p = torch.softmax(x, dim=1)
        mean = self.mean_conv(p)
        l = None
        if self.type == 'l1':
            l = torch.abs(p - mean).sum(dim=1)
        elif self.type == 'kl':
            l = torch.sum(p * torch.log(p / (mean + 1e-06) + 1e-06), dim=1)
        else:
            raise NotImplementedError('not implemented local soft loss: {}'
                .format(self.type))
        return l


def get_inputs():
    return [torch.rand([4, 19, 4, 4])]


def get_init_inputs():
    return [[], {}]
