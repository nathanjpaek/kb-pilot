import torch
import torch.nn as nn


class GatedPooling1(nn.Module):
    """
        Gated pooling as defined in https://arxiv.org/abs/1509.08985
        This implementation is the L variant ( entire layer, one parameter )
    """

    def __init__(self, kernel_size):
        super(GatedPooling1, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size)
        self.maxpool = nn.MaxPool2d(kernel_size)
        self.transform = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=
            kernel_size)
        torch.nn.init.kaiming_normal_(self.transform.weight)

    def forward(self, x):
        xs = [self.transform(x_filt.unsqueeze(1)).squeeze(1) for x_filt in
            torch.unbind(x, dim=1)]
        alpha = torch.sigmoid(torch.stack(xs, 1))
        return alpha * self.maxpool(x) + (1 - alpha) * self.avgpool(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4}]
