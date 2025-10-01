import torch
import numpy as np
import torch.nn.functional as F
from torch import nn


class AEC(nn.Module):

    def __init__(self, hidden_nodes, conv_width, pixel_patchsize,
        lambda_activation):
        super(AEC, self).__init__()
        self.hidden_nodes = hidden_nodes
        self.conv_width = conv_width
        self.pixel_patchsize = pixel_patchsize
        self.temporal_conv_kernel_size = (conv_width, pixel_patchsize,
            pixel_patchsize)
        self.lambda_activation = lambda_activation
        self.tconv = nn.utils.weight_norm(nn.Conv3d(1, hidden_nodes,
            kernel_size=self.temporal_conv_kernel_size, stride=1), dim=0,
            name='weight')
        self.tdeconv = nn.ConvTranspose3d(hidden_nodes, 1, kernel_size=np.
            transpose(self.temporal_conv_kernel_size), stride=1)
        torch.nn.init.normal_(self.tconv.weight, mean=0, std=1)

    def format_data_numpy(self, x):
        x = np.detach().cpu().numpy()
        return x

    def encode(self, x):
        noise = 0
        return F.relu(self.tconv(x + noise))

    def decode(self, z):
        return self.tdeconv(z)

    def forward(self, x):
        activations = self.encode(x)
        decoded = self.decode(activations)
        return activations, decoded

    def loss_func(self, x, xhat, activations):
        recon_loss = nn.MSELoss()(xhat, x)
        mean_activation = activations.mean((0, 2))
        goal_activation = torch.ones_like(mean_activation)
        activation_loss = torch.abs(mean_activation - goal_activation).mean()
        loss = recon_loss + 1000 * activation_loss
        return loss

    def calc_snr(self, x, xhat):
        """
        Calculate the ssignal to noise ratio for a reconstructed signal
        Params:
            x(Array): The signal before transformation
            x_hat (Array): The signal after transformation. Must be of same type of s.
        Returns:
            snr (float): The signal to noise ratio of the transformed signal s_prime
        """
        x = x.flatten()
        xhat = x.flatten()
        signal = x.mean()
        noise = (xhat - x).std()
        snr = 10 * (signal / noise).log10()
        return snr


def get_inputs():
    return [torch.rand([4, 1, 64, 64, 64])]


def get_init_inputs():
    return [[], {'hidden_nodes': 4, 'conv_width': 4, 'pixel_patchsize': 4,
        'lambda_activation': 4}]
