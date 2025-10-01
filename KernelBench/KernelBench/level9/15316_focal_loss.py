import torch
import torch.nn as nn


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t = t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float(
        ) * t_max
    return result


class focal_loss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0, size_average=True):
        super(focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, gt):
        gt_oh = torch.cat((gt, 1.0 - gt), dim=1)
        pt = (gt_oh * pred).sum(1)
        focal_map = -self.alpha * torch.pow(1.0 - pt, self.gamma) * torch.log2(
            clip_by_tensor(pt, 1e-12, 1.0))
        if self.size_average:
            loss = focal_map.mean()
        else:
            loss = focal_map.sum()
        return loss


def get_inputs():
    return [torch.rand([4, 8, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
