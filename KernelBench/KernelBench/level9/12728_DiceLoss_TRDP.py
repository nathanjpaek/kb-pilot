import torch
from torch.nn.modules.loss import _Loss


class DiceLoss_TRDP(_Loss):

    def __init__(self, per_image=False):
        super(DiceLoss_TRDP, self).__init__()
        self.per_image = per_image

    def forward(self, y_pred, y_true):
        """
        :param y_pred: NxCxHxW
        :param y_true: NxCxHxW
        :return: scalar
        """
        per_image = self.per_image
        y_pred = y_pred.sigmoid()
        batch_size = y_pred.size()[0]
        eps = 1e-05
        if not per_image:
            batch_size = 1
        dice_target = y_true.contiguous().view(batch_size, -1).float()
        dice_output = y_pred.contiguous().view(batch_size, -1)
        intersection = torch.sum(dice_output * dice_target, dim=1)
        union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1
            ) + eps
        loss = (1 - (2 * intersection + eps) / union).mean()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
