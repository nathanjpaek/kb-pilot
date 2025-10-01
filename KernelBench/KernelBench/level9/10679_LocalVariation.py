import torch
import torch.nn as nn


class LocalVariation(nn.Module):
    """Layer to compute the LocalVariation  of an image
    """

    def __init__(self, k_size=5):
        super(LocalVariation, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(k_size, 1)
        self.mu_y_pool = nn.AvgPool2d(k_size, 1)
        self.sig_x_pool = nn.AvgPool2d(k_size, 1)
        self.sig_y_pool = nn.AvgPool2d(k_size, 1)
        self.sig_xy_pool = nn.AvgPool2d(k_size, 1)
        self.refl = nn.ReflectionPad2d(k_size // 2)

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_x + sigma_y
        return sigma_y


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
