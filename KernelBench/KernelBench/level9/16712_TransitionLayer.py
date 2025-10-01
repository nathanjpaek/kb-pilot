import torch
import torch.nn as nn
import torch.nn.functional as F


class TransitionLayer(nn.Module):
    """ TransitionLayer between dense blocks
    """

    def __init__(self, n_in, n_out, use_dropout=False):
        """
        Args:
            n_in (int) : number of input channels
            n_out (int) : number of output channels
            use_dropout (bool) : whether use dropout after conv layer
        """
        super(TransitionLayer, self).__init__()
        self.conv1x1 = nn.Conv2d(n_in, n_out, 1)
        self.use_dropout = use_dropout

    def forward(self, x):
        x = self.conv1x1(x)
        if self.use_dropout:
            x = F.dropout(x, p=0.2, training=self.training)
        x = F.avg_pool2d(x, 2)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_in': 4, 'n_out': 4}]
