import torch
from torch import nn


def cosine_sim(x1, x2, dim=1, eps=1e-08):
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1, w2).clamp(min=eps)


class LargeMarginCosLoss(nn.Module):
    """
    CosFace: Large Margin Cosine Loss for Deep Face Recognition. CVPR2018
    """

    def __init__(self, feature_size, class_num, s=7.0, m=0.2):
        super(LargeMarginCosLoss, self).__init__()
        self.feature_size = feature_size
        self.class_num = class_num
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.randn(class_num, feature_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label):
        cosine = cosine_sim(x, self.weight)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = self.s * (cosine - one_hot * self.m)
        return output, cosine


def get_inputs():
    return [torch.rand([4, 4]), torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {'feature_size': 4, 'class_num': 4}]
