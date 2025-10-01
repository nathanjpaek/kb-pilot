from torch.nn import Module
import math
import torch
from torch.nn import functional as F
import torch._utils
from torch.nn import Parameter
from itertools import product as product
import torch.utils.data.distributed


class MV_Softmax(Module):
    """Implementation for "Mis-classified Vector Guided Softmax Loss for Face Recognition"
    """

    def __init__(self, feat_dim, num_class, is_am, margin=0.35, mv_weight=
        1.12, scale=32):
        super(MV_Softmax, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_class))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-05).mul_(100000.0)
        self.margin = margin
        self.mv_weight = mv_weight
        self.scale = scale
        self.is_am = is_am
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = self.sin_m * margin

    def forward(self, x, label):
        kernel_norm = F.normalize(self.weight, dim=0)
        cos_theta = torch.mm(x, kernel_norm)
        batch_size = label.size(0)
        gt = cos_theta[torch.arange(0, batch_size), label].view(-1, 1)
        if self.is_am:
            mask = cos_theta > gt - self.margin
            final_gt = torch.where(gt > self.margin, gt - self.margin, gt)
        else:
            sin_theta = torch.sqrt(1.0 - torch.pow(gt, 2))
            cos_theta_m = gt * self.cos_m - sin_theta * self.sin_m
            mask = cos_theta > cos_theta_m
            final_gt = torch.where(gt > 0.0, cos_theta_m, gt)
        hard_example = cos_theta[mask]
        cos_theta[mask] = self.mv_weight * hard_example + self.mv_weight - 1.0
        cos_theta.scatter_(1, label.data.view(-1, 1), final_gt)
        cos_theta *= self.scale
        return cos_theta


def get_inputs():
    return [torch.rand([4, 4]), torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {'feat_dim': 4, 'num_class': 4, 'is_am': 4}]
