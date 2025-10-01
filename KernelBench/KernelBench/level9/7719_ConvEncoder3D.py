import torch
from torch import nn


class ConvEncoder3D(nn.Module):
    """ Simple convolutional conditioning network.

    It consists of 6 convolutional layers, each downsampling the input by a
    factor of 2, and a final fully-connected layer projecting the output to
    c_dim dimensions.
    """

    def __init__(self, c_dim=128, hidden_dim=32, **kwargs):
        """ Initialisation.

        Args:
            c_dim (int): output dimension of the latent embedding
        """
        super().__init__()
        self.conv0 = nn.Conv3d(3, hidden_dim, 3, stride=(1, 2, 2), padding=1)
        self.conv1 = nn.Conv3d(hidden_dim, hidden_dim * 2, 3, stride=(2, 2,
            2), padding=1)
        self.conv2 = nn.Conv3d(hidden_dim * 2, hidden_dim * 4, 3, stride=(1,
            2, 2), padding=1)
        self.conv3 = nn.Conv3d(hidden_dim * 4, hidden_dim * 8, 3, stride=(2,
            2, 2), padding=1)
        self.conv4 = nn.Conv3d(hidden_dim * 8, hidden_dim * 16, 3, stride=(
            2, 2, 2), padding=1)
        self.conv5 = nn.Conv3d(hidden_dim * 16, hidden_dim * 16, 3, stride=
            (2, 2, 2), padding=1)
        self.fc_out = nn.Linear(hidden_dim * 16, c_dim)
        self.actvn = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        batch_size = x.size(0)
        net = self.conv0(x)
        net = self.conv1(self.actvn(net))
        net = self.conv2(self.actvn(net))
        net = self.conv3(self.actvn(net))
        net = self.conv4(self.actvn(net))
        net = self.conv5(self.actvn(net))
        final_dim = net.shape[1]
        net = net.view(batch_size, final_dim, -1).mean(2)
        out = self.fc_out(self.actvn(net))
        return out


def get_inputs():
    return [torch.rand([4, 3, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
