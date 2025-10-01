import torch
import torch.nn as nn


def reduction_batch_based(image_loss, M):
    divisor = torch.sum(M)
    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):
    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))
    return reduction(image_loss, 2 * M)


def reduction_image_based(image_loss, M):
    valid = M.nonzero()
    image_loss[valid] = image_loss[valid] / M[valid]
    return torch.mean(image_loss)


class MSELoss(nn.Module):

    def __init__(self, reduction='batch-based'):
        super().__init__()
        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
