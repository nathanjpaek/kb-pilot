import torch
import torch.nn as nn
import torch.utils.data


class Quaternion(nn.Module):

    def __init__(self):
        super(Quaternion, self).__init__()

    def forward(self, rvec):
        theta = torch.sqrt(1e-05 + torch.sum(rvec ** 2, dim=1))
        rvec = rvec / theta[:, None]
        return torch.stack((1.0 - 2.0 * rvec[:, 1] ** 2 - 2.0 * rvec[:, 2] **
            2, 2.0 * (rvec[:, 0] * rvec[:, 1] - rvec[:, 2] * rvec[:, 3]), 
            2.0 * (rvec[:, 0] * rvec[:, 2] + rvec[:, 1] * rvec[:, 3]), 2.0 *
            (rvec[:, 0] * rvec[:, 1] + rvec[:, 2] * rvec[:, 3]), 1.0 - 2.0 *
            rvec[:, 0] ** 2 - 2.0 * rvec[:, 2] ** 2, 2.0 * (rvec[:, 1] *
            rvec[:, 2] - rvec[:, 0] * rvec[:, 3]), 2.0 * (rvec[:, 0] * rvec
            [:, 2] - rvec[:, 1] * rvec[:, 3]), 2.0 * (rvec[:, 0] * rvec[:, 
            3] + rvec[:, 1] * rvec[:, 2]), 1.0 - 2.0 * rvec[:, 0] ** 2 - 
            2.0 * rvec[:, 1] ** 2), dim=1).view(-1, 3, 3)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
