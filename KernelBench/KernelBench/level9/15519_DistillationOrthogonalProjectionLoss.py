import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class DistillationOrthogonalProjectionLoss(nn.Module):

    def __init__(self):
        super(DistillationOrthogonalProjectionLoss, self).__init__()

    @staticmethod
    def forward(features, features_teacher):
        features = F.normalize(features, p=2, dim=1)
        features_teacher = F.normalize(features_teacher, p=2, dim=1)
        dot_prod = torch.matmul(features, features.t())
        dot_prod_teacher = torch.matmul(features_teacher, features_teacher.t())
        tau = 1
        loss = F.kl_div(dot_prod / tau, dot_prod_teacher / tau, reduction=
            'sum', log_target=True) * (tau * tau) / dot_prod_teacher.numel()
        return loss


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
