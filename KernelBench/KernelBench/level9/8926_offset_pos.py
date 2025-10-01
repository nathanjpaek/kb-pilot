import torch
import torch.nn as nn


class offset_pos(nn.Module):

    def __init__(self):
        super(offset_pos, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, offset_pred, offset_label):
        l1_loss = offset_label[:, 2, :, :].unsqueeze(dim=1) * self.smoothl1(
            offset_pred, offset_label[:, :2, :, :])
        off_loss = torch.sum(l1_loss) / max(1.0, torch.sum(offset_label[:, 
            2, :, :]))
        return off_loss


def get_inputs():
    return [torch.rand([4, 2, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
