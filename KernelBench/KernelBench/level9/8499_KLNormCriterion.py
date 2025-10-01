import torch
import torch.nn as nn


class KLNormCriterion(nn.Module):

    def __init__(self):
        super(KLNormCriterion, self).__init__()

    def forward(self, z_mean_pre, z_log_sigma_pre, z_mean_gt=None,
        z_sigma_gt=None):
        batch_size = z_mean_pre.size(0)
        if z_mean_gt is None or z_sigma_gt is None:
            """
            KL[N(z_mean_pre,z_sigma_pre)||N(0,I)]
            """
            z_mean_sq = z_mean_pre * z_mean_pre
            z_log_sigma_sq = 2 * z_log_sigma_pre
            z_sigma_sq = torch.exp(z_log_sigma_sq)
            kl_loss = 0.5 * torch.sum(z_mean_sq + z_sigma_sq -
                z_log_sigma_sq - 1) / batch_size
        else:
            """
            KL[N(z_mean_pre,z_sigma_pre)||N(z_mean_gt,z_sigma_gt)]
            """
            z_log_sigma_sq_pre = 2 * z_log_sigma_pre
            z_sigma_sq_pre = torch.exp(z_log_sigma_sq_pre)
            z_log_sigma_sq_gt = 2 * torch.log(z_sigma_gt + 0.0001)
            z_sigma_sq_gt = z_sigma_gt ** 2
            kl_loss = 0.5 * torch.sum(z_log_sigma_sq_gt -
                z_log_sigma_sq_pre + z_sigma_sq_pre / z_sigma_sq_gt + (
                z_mean_pre - z_mean_gt) ** 2 / z_sigma_sq_gt - 1) / batch_size
        return kl_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
