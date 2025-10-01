import torch
import torch.nn as nn


class TotalVariationLoss(nn.Module):

    def __init__(self, loss_weight: 'int'=1) ->None:
        super(TotalVariationLoss, self).__init__()
        self.loss_weight = loss_weight

    @staticmethod
    def tensor_size(t: 'torch.Tensor') ->torch.Tensor:
        return t.size()[1] * t.size()[2] * t.size()[3]

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        batch_size = x.size()[0]
        h = x.size()[2]
        w = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w - 1], 2).sum()
        tv_loss = self.loss_weight * 2 * (h_tv / count_h + w_tv / count_w
            ) / batch_size
        return tv_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
