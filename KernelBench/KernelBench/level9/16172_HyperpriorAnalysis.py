import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperpriorAnalysis(nn.Module):
    """
    Hyperprior 'analysis model' as proposed in [1]. 

    [1] Ballé et. al., "Variational image compression with a scale hyperprior", 
        arXiv:1802.01436 (2018).

    C:  Number of input channels
    """

    def __init__(self, C=220, N=320, activation='relu'):
        super(HyperpriorAnalysis, self).__init__()
        cnn_kwargs = dict(kernel_size=5, stride=2, padding=2, padding_mode=
            'reflect')
        self.activation = getattr(F, activation)
        self.n_downsampling_layers = 2
        self.conv1 = nn.Conv2d(C, N, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(N, N, **cnn_kwargs)
        self.conv3 = nn.Conv2d(N, N, **cnn_kwargs)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)
        return x


def get_inputs():
    return [torch.rand([4, 220, 64, 64])]


def get_init_inputs():
    return [[], {}]
