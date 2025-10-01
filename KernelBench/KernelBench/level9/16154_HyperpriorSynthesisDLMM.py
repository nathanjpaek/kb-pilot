import torch
import torch.nn as nn
import torch.nn.functional as F


def get_num_DLMM_channels(C, K=4, params=['mu', 'scale', 'mix']):
    """
    C:  Channels of latent representation (L3C uses 5).
    K:  Number of mixture coefficients.
    """
    return C * K * len(params)


class HyperpriorSynthesisDLMM(nn.Module):
    """
    Outputs distribution parameters of input latents, conditional on 
    hyperlatents, assuming a discrete logistic mixture model.

    C:  Number of output channels
    """

    def __init__(self, C=64, N=320, activation='relu', final_activation=None):
        super(HyperpriorSynthesisDLMM, self).__init__()
        cnn_kwargs = dict(kernel_size=5, stride=2, padding=2, output_padding=1)
        self.activation = getattr(F, activation)
        self.final_activation = final_activation
        self.conv1 = nn.ConvTranspose2d(N, N, **cnn_kwargs)
        self.conv2 = nn.ConvTranspose2d(N, N, **cnn_kwargs)
        self.conv3 = nn.ConvTranspose2d(N, C, kernel_size=3, stride=1,
            padding=1)
        self.conv_out = nn.Conv2d(C, get_num_DLMM_channels(C), kernel_size=
            1, stride=1)
        if self.final_activation is not None:
            self.final_activation = getattr(F, final_activation)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)
        x = self.conv_out(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x


def get_inputs():
    return [torch.rand([4, 320, 4, 4])]


def get_init_inputs():
    return [[], {}]
