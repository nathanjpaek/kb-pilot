import torch
import torch.nn as nn


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = std.data.new(std.size()).normal_()
    return mu + std * eps


class Encoder_H(nn.Module):

    def __init__(self, input_shape=(64, 64), z_dim=10, nc=3, padding=1):
        super(Encoder_H, self).__init__()
        self.conv2d_1 = nn.Conv2d(nc, 32, 4, 2, padding)
        self.conv2d_2 = nn.Conv2d(32, 32, 4, 2, padding)
        self.conv2d_3 = nn.Conv2d(32, 64, 4, 2, padding)
        self.conv2d_4 = nn.Conv2d(64, 64, 4, 2, padding)
        self.flatten_shape, self.dconv_size = self._get_conv_output(input_shape
            , nc)
        self.linear = nn.Linear(self.flatten_shape, z_dim * 2)

    def _get_conv_output(self, shape, nc):
        bs = 1
        dummy_x = torch.empty(bs, nc, *shape)
        x, dconv_size = self._forward_features(dummy_x)
        flatten_shape = x.flatten(1).size(1)
        return flatten_shape, dconv_size

    def _forward_features(self, x):
        size0 = x.shape[1:]
        x = torch.relu(self.conv2d_1(x))
        size1 = x.shape[1:]
        x = torch.relu(self.conv2d_2(x))
        size2 = x.shape[1:]
        x = torch.relu(self.conv2d_3(x))
        size3 = x.shape[1:]
        x = torch.relu(self.conv2d_4(x))
        size4 = x.shape[1:]
        return x, [size0, size1, size2, size3, size4]

    def forward(self, x):
        x = torch.relu(self.conv2d_1(x))
        x = torch.relu(self.conv2d_2(x))
        x = torch.relu(self.conv2d_3(x))
        x = torch.relu(self.conv2d_4(x))
        x = self.linear(x.flatten(1))
        return x


class Decoder_H(nn.Module):

    def __init__(self, output_shape, z_dim=10, nc=3, padding=1):
        super(Decoder_H, self).__init__()
        self.output_shape = output_shape
        flatten_shape = output_shape[-1][0] * output_shape[-1][1
            ] * output_shape[-1][2]
        self.linear = nn.Linear(z_dim, flatten_shape)
        self.conv2d_1 = nn.ConvTranspose2d(64, 64, 4, 2, padding)
        self.conv2d_2 = nn.ConvTranspose2d(64, 32, 4, 2, padding)
        self.conv2d_3 = nn.ConvTranspose2d(32, 32, 4, 2, padding)
        self.conv2d_4 = nn.ConvTranspose2d(32, nc, 4, 2, padding)

    def _forward_features(self, x):
        x = torch.relu(self.conv2d_1(x, self.output_shape[3][1:]))
        x = torch.relu(self.conv2d_2(x, self.output_shape[2][1:]))
        x = torch.relu(self.conv2d_3(x, self.output_shape[1][1:]))
        x = self.conv2d_4(x, self.output_shape[0][1:])
        return x

    def forward(self, x):
        x = torch.relu(self.linear(x))
        x = x.view(-1, *self.output_shape[4])
        x = self._forward_features(x)
        return x


class BetaVAE_H(nn.Module):

    def __init__(self, input_shape=(64, 64), z_dim=10, nc=3, padding=0,
        activation=nn.Identity()):
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.activation = activation
        self.encoder = Encoder_H(input_shape=input_shape, nc=nc, z_dim=
            z_dim, padding=padding)
        self.decoder = Decoder_H(self.encoder.dconv_size, nc=nc, z_dim=
            z_dim, padding=padding)

    def forward(self, x):
        distributions = self.encoder(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self.decoder(z)
        return self.activation(x_recon), mu, logvar


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
