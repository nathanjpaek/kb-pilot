import torch
import torch.nn as nn
import torch.nn.functional as F


class KD(nn.Module):

    def __init__(self, alpha, T):
        super(KD, self).__init__()
        self.alpha = alpha
        self.T = T

    def forward(self, output_stu, output_tch, label):
        loss_stu = F.cross_entropy(output_stu, label)
        label_stu = F.log_softmax(output_stu / self.T, dim=1)
        label_tch = F.softmax(output_tch / self.T, dim=1)
        loss_tch = F.kl_div(label_stu, label_tch) * self.T * self.T
        loss = loss_stu * (1 - self.alpha) + loss_tch * self.alpha
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'alpha': 4, 'T': 4}]
