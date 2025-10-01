import torch
import torch.nn as nn


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)


def weights_init(m):
    classname = m.__class__.__name__
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        m.weight.data.normal_(0.0, 0.02)
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0)


class Noise_injector(nn.Module):

    def __init__(self, n_hidden, z_dim, num_channels, n_channels_out,
        device='cpu'):
        super(Noise_injector, self).__init__()
        self.num_channels = num_channels
        self.n_channels_out = n_channels_out
        self.n_hidden = n_hidden
        self.z_dim = z_dim
        self.device = device
        self.residual = nn.Linear(self.z_dim, self.n_hidden)
        self.scale = nn.Linear(self.z_dim, self.n_hidden)
        self.last_layer = nn.Conv2d(self.n_hidden, self.n_channels_out,
            kernel_size=1)
        self.residual.apply(weights_init)
        self.scale.apply(weights_init)
        self.last_layer.apply(init_weights_orthogonal_normal)

    def forward(self, feature_map, z):
        """
        Z is B x Z_dim and feature_map is B x C x H x W.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        residual = self.residual(z).view(z.shape[0], self.n_hidden, 1, 1)
        scale = self.scale(z).view(z.shape[0], self.n_hidden, 1, 1)
        feature_map = (feature_map + residual) * (scale + 1e-05)
        return self.last_layer(feature_map)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'n_hidden': 4, 'z_dim': 4, 'num_channels': 4,
        'n_channels_out': 4}]
