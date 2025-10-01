import torch
import torch.nn
import torch.nn as nn
import torch.nn.parallel


class SpatialAttention2d(nn.Module):
    """
    SpatialAttention2d
    2-layer 1x1 conv network with softplus activation.
    <!!!> attention score normalization will be added for experiment.
    """

    def __init__(self, in_c, act_fn='relu'):
        super(SpatialAttention2d, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 512, 1, 1)
        if act_fn.lower() in ['relu']:
            self.act1 = nn.ReLU()
        elif act_fn.lower() in ['leakyrelu', 'leaky', 'leaky_relu']:
            self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(512, 1, 1, 1)
        self.softplus = nn.Softplus(beta=1, threshold=20)

    def forward(self, x):
        """
        x : spatial feature map. (b x c x w x h)
        s : softplus attention score 
        """
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.softplus(x)
        return x

    def __repr__(self):
        return self.__class__.__name__


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_c': 4}]
