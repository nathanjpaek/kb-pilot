import torch
import numpy as np
import torch.nn as nn
import torch.utils.data


class WPMLoss(nn.Module):

    def __init__(self, weight):
        super(WPMLoss, self).__init__()
        self.weight = weight

    def forward(self, y_real, y_imag, y_real_hat, y_imag_hat):
        torch.FloatTensor([np.pi])
        mag = torch.sqrt(y_real ** 2 + y_imag ** 2)
        mag_hat = torch.sqrt(y_real_hat ** 2 + y_imag_hat ** 2)
        theta = torch.atan2(y_imag, y_real)
        theta_hat = torch.atan2(y_imag_hat, y_real_hat)
        dif_theta = 2 * mag * torch.sin((theta_hat - theta) / 2)
        dif_mag = mag_hat - mag
        loss = torch.mean(dif_mag ** 2 + self.weight * dif_theta ** 2)
        if torch.isnan(loss).any():
            np.save('y_real.npy', y_real.data.cpu().numpy())
            np.save('y_imag.npy', y_imag.data.cpu().numpy())
            np.save('y_real_hat.npy', y_real_hat.data.cpu().numpy())
            np.save('y_imag_hat.npy', y_imag_hat.data.cpu().numpy())
            raise ValueError('NAN encountered in loss')
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'weight': 4}]
