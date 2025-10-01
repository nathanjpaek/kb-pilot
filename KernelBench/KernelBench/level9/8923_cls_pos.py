import torch
import torch.nn as nn


class cls_pos(nn.Module):

    def __init__(self):
        super(cls_pos, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pos_pred, pos_label):
        log_loss = self.bce(pos_pred[:, 0, :, :], pos_label[:, 2, :, :])
        pos_pred = pos_pred.sigmoid()
        positives = pos_label[:, 2, :, :]
        negatives = pos_label[:, 1, :, :] - pos_label[:, 2, :, :]
        fore_weight = positives * (1.0 - pos_pred[:, 0, :, :]) ** 2
        back_weight = negatives * (1.0 - pos_label[:, 0, :, :]
            ) ** 4.0 * pos_pred[:, 0, :, :] ** 2.0
        focal_weight = fore_weight + back_weight
        assigned_box = torch.sum(pos_label[:, 2, :, :])
        cls_loss = torch.sum(focal_weight * log_loss) / max(1.0, assigned_box)
        return cls_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
