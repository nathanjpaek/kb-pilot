import torch
import torch.nn as nn
import torch.utils.data


class Rodrigues(nn.Module):

    def __init__(self):
        super(Rodrigues, self).__init__()

    def forward(self, rvec):
        theta = torch.sqrt(1e-05 + torch.sum(rvec ** 2, dim=1))
        rvec = rvec / theta[:, None]
        costh = torch.cos(theta)
        sinth = torch.sin(theta)
        return torch.stack((rvec[:, 0] ** 2 + (1.0 - rvec[:, 0] ** 2) *
            costh, rvec[:, 0] * rvec[:, 1] * (1.0 - costh) - rvec[:, 2] *
            sinth, rvec[:, 0] * rvec[:, 2] * (1.0 - costh) + rvec[:, 1] *
            sinth, rvec[:, 0] * rvec[:, 1] * (1.0 - costh) + rvec[:, 2] *
            sinth, rvec[:, 1] ** 2 + (1.0 - rvec[:, 1] ** 2) * costh, rvec[
            :, 1] * rvec[:, 2] * (1.0 - costh) - rvec[:, 0] * sinth, rvec[:,
            0] * rvec[:, 2] * (1.0 - costh) - rvec[:, 1] * sinth, rvec[:, 1
            ] * rvec[:, 2] * (1.0 - costh) + rvec[:, 0] * sinth, rvec[:, 2] **
            2 + (1.0 - rvec[:, 2] ** 2) * costh), dim=1).view(-1, 3, 3)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
