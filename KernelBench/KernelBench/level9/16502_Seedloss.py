import torch
import torch.nn as nn
from torch.nn import functional as F


class Seedloss(nn.Module):

    def __init__(self, ignore_label=21):
        super(Seedloss, self).__init__()
        self.ignore_label = ignore_label
        self.eps = 1e-05

    def my_softmax(self, score, dim=1):
        probs = torch.clamp(F.softmax(score, dim), self.eps, 1)
        probs = probs / torch.sum(probs, dim=dim, keepdim=True)
        return probs

    def forward(self, predict, target):
        """
        compute balanced seed loss
        :param predict: (n, c, h, w)
        :param target: (n, c, h, w)
        :return:
        """
        assert not target.requires_grad
        target = target
        assert torch.sum(torch.isinf(predict)) == 0
        assert torch.sum(torch.isnan(predict)) == 0
        input_log_prob = torch.log(self.my_softmax(predict, dim=1))
        assert torch.sum(torch.isnan(input_log_prob)) == 0
        fg_prob = input_log_prob[:, 1:, :, :]
        fg_label = target[:, 1:, :, :]
        fg_count = torch.sum(fg_label, dim=(1, 2, 3)) + self.eps
        bg_prob = input_log_prob[:, 0:1, :, :]
        bg_label = target[:, 0:1, :, :]
        bg_count = torch.sum(bg_label, dim=(1, 2, 3)) + self.eps
        loss_fg = torch.sum(fg_label * fg_prob, dim=(1, 2, 3))
        loss_fg = -1 * torch.mean(loss_fg / fg_count)
        loss_bg = torch.sum(bg_label * bg_prob, dim=(1, 2, 3))
        loss_bg = -1 * torch.mean(loss_bg / bg_count)
        total_loss = loss_bg + loss_fg
        assert torch.sum(torch.isnan(total_loss)
            ) == 0, 'fg_loss: {} fg_count: {} bg_loss: {} bg_count: {}'.format(
            loss_fg, fg_count, loss_bg, bg_count)
        return total_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
