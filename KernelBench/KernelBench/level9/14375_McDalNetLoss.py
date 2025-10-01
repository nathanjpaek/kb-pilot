import torch
import torch.nn as nn
import torch.nn.functional as F


def discrepancy_slice_wasserstein(p1, p2):
    s = p1.shape
    if s[1] > 1:
        proj = torch.randn(s[1], 128)
        proj *= torch.rsqrt(torch.sum(torch.mul(proj, proj), 0, keepdim=True))
        p1 = torch.matmul(p1, proj)
        p2 = torch.matmul(p2, proj)
    p1 = torch.topk(p1, s[0], dim=0)[0]
    p2 = torch.topk(p2, s[0], dim=0)[0]
    dist = p1 - p2
    wdist = torch.mean(torch.mul(dist, dist))
    return wdist


class _Loss(nn.Module):

    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average


class _WeightedLoss(_Loss):

    def __init__(self, weight=None, size_average=True):
        super(_WeightedLoss, self).__init__(size_average)
        self.register_buffer('weight', weight)


class McDalNetLoss(_WeightedLoss):

    def __init__(self, weight=None, size_average=True):
        super(McDalNetLoss, self).__init__(weight, size_average)

    def forward(self, input1, input2, dis_type='L1'):
        if dis_type == 'L1':
            prob_s = F.softmax(input1, dim=1)
            prob_t = F.softmax(input2, dim=1)
            loss = torch.mean(torch.abs(prob_s - prob_t))
        elif dis_type == 'CE':
            loss = -F.log_softmax(input2, dim=1).mul(F.softmax(input1, dim=1)
                ).mean() - F.log_softmax(input1, dim=1).mul(F.softmax(
                input2, dim=1)).mean()
            loss = loss * 0.5
        elif dis_type == 'KL':
            loss = F.kl_div(F.log_softmax(input1), F.softmax(input2)
                ) + F.kl_div(F.log_softmax(input2), F.softmax(input1))
            loss = loss * 0.5
        elif dis_type == 'L2':
            nClass = input1.size()[1]
            prob_s = F.softmax(input1, dim=1)
            prob_t = F.softmax(input2, dim=1)
            loss = torch.norm(prob_s - prob_t, p=2, dim=1).mean() / nClass
        elif dis_type == 'Wasse':
            prob_s = F.softmax(input1, dim=1)
            prob_t = F.softmax(input2, dim=1)
            loss = discrepancy_slice_wasserstein(prob_s, prob_t)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
