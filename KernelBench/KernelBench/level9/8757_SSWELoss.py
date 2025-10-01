import torch
import torch.nn as nn


class HingeMarginLoss(nn.Module):
    """
    计算hinge loss 接口
    """

    def __init__(self):
        super(HingeMarginLoss, self).__init__()

    def forward(self, t, tr, delt=None, size_average=False):
        """
        计算hingle loss
        """
        if delt is None:
            loss = torch.clamp(1 - t + tr, min=0)
        else:
            loss = torch.clamp(1 - torch.mul(t - tr, torch.squeeze(delt)),
                min=0)
            loss = torch.unsqueeze(loss, dim=-1)
        return loss


class SSWELoss(nn.Module):
    """
    SSWE 中考虑gram得分和情感得分的loss类
    """

    def __init__(self, alpha=0.5):
        super(SSWELoss, self).__init__()
        self.alpha = alpha
        self.hingeLoss = HingeMarginLoss()

    def forward(self, scores, labels, size_average=False):
        """
        [(true_sy_score,true_sem_score), (corrput_sy_score,corrupt_sem_score),...]
        """
        assert len(scores) >= 2
        true_score = scores[0]
        sem_loss = self.hingeLoss(true_score[1][:, 0], true_score[1][:, 1],
            delt=labels)
        loss = []
        for corpt_i in range(1, len(scores)):
            syn_loss = self.hingeLoss(true_score[0], scores[corpt_i][0])
            cur_loss = syn_loss * self.alpha + sem_loss * (1 - self.alpha)
            loss.append(cur_loss)
        loss = torch.cat(loss, dim=0)
        if size_average:
            loss = torch.mean(loss)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
