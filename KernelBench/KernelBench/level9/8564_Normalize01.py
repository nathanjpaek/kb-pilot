import torch
import torch.nn as nn


class Normalize01(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, result_noisy):
        Nbatch = result_noisy.size(0)
        result_noisy_01 = torch.zeros_like(result_noisy)
        for i in range(Nbatch):
            min_val = result_noisy[i, :, :, :].min()
            max_val = result_noisy[i, :, :, :].max()
            result_noisy_01[i, :, :, :] = (result_noisy[i, :, :, :] - min_val
                ) / (max_val - min_val)
        return result_noisy_01


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
