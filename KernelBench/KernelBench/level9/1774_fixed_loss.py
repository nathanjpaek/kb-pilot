import torch
import torch.nn as nn
import torch.nn.functional as F


class fixed_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, out_image, gt_image, est_noise, gt_noise, if_asym):
        h_x = est_noise.size()[2]
        w_x = est_noise.size()[3]
        count_h = self._tensor_size(est_noise[:, :, 1:, :])
        count_w = self._tensor_size(est_noise[:, :, :, 1:])
        h_tv = torch.pow(est_noise[:, :, 1:, :] - est_noise[:, :, :h_x - 1,
            :], 2).sum()
        w_tv = torch.pow(est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x -
            1], 2).sum()
        tvloss = h_tv / count_h + w_tv / count_w
        loss = torch.mean(torch.pow(out_image - gt_image, 2)
            ) + if_asym * 0.5 * torch.mean(torch.mul(torch.abs(0.3 - F.relu
            (gt_noise - est_noise)), torch.pow(est_noise - gt_noise, 2))
            ) + 0.05 * tvloss
        return loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
