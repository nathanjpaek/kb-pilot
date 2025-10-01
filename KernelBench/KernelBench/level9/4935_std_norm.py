import torch
import torch.nn as nn


class std_norm(nn.Module):

    def __init__(self, inverse=False):
        super(std_norm, self).__init__()
        self.inverse = inverse

    def forward(self, x, mean, std):
        out = []
        for i in range(len(mean)):
            if not self.inverse:
                normalized = (x[:, i, :, :] - mean[i]) / std[i]
            else:
                normalized = x[:, i, :, :] * std[i] + mean[i]
            normalized = torch.unsqueeze(normalized, 1)
            out.append(normalized)
        return torch.cat(out, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
