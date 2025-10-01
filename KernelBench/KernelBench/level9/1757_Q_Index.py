import torch
import torch.nn as nn


class Q_Index(nn.Module):
    """
    Quality measurement between perturbated (image with applied noise) and denoised target image.
    This module works only for images with a single color channel (grayscale)
    """

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        batch_size = input.shape[0]
        input = input.view(batch_size, -1)
        target = target.view(batch_size, -1)
        input_mean = input.mean(dim=-1)
        target_mean = target.mean(dim=-1)
        input_var = input.var(dim=-1)
        target_var = target.var(dim=-1)
        mean_inp_times_tar = torch.mean(input * target, dim=-1)
        covariance = mean_inp_times_tar - input_mean * target_mean
        Q = 4.0 * covariance * input_mean * target_mean / ((input_var +
            target_var) * (input_mean ** 2 + target_mean ** 2))
        Q = Q.mean()
        return Q


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
