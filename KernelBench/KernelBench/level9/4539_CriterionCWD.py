import torch
import torch.nn as nn
import torch._utils
import torch.optim


class ChannelNorm(nn.Module):

    def __init__(self):
        super(ChannelNorm, self).__init__()

    def forward(self, featmap):
        n, c, _h, _w = featmap.shape
        featmap = featmap.reshape((n, c, -1))
        featmap = featmap.softmax(dim=-1)
        return featmap


class CriterionCWD(nn.Module):

    def __init__(self, norm_type='channel', divergence='kl', temperature=4.0):
        super(CriterionCWD, self).__init__()
        if norm_type == 'channel':
            self.normalize = ChannelNorm()
        elif norm_type == 'spatial':
            self.normalize = nn.Softmax(dim=1)
        elif norm_type == 'channel_mean':
            self.normalize = lambda x: x.view(x.size(0), x.size(1), -1).mean(-1
                )
        else:
            self.normalize = None
        self.norm_type = norm_type
        self.temperature = 1.0
        if divergence == 'mse':
            self.criterion = nn.MSELoss(reduction='sum')
        elif divergence == 'kl':
            self.criterion = nn.KLDivLoss(reduction='sum')
            self.temperature = temperature
        self.divergence = divergence

    def forward(self, preds_S, preds_T):
        n, c, h, w = preds_S.shape
        if self.normalize is not None:
            norm_s = self.normalize(preds_S / self.temperature)
            norm_t = self.normalize(preds_T.detach() / self.temperature)
        else:
            norm_s = preds_S
            norm_t = preds_T.detach()
        if self.divergence == 'kl':
            norm_s = norm_s.log()
        loss = self.criterion(norm_s, norm_t)
        if self.norm_type == 'channel' or self.norm_type == 'channel_mean':
            loss /= n * c
        else:
            loss /= n * h * w
        return loss * self.temperature ** 2


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
