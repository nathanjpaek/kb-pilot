import torch
import torch.nn as nn


class reg_pos(nn.Module):

    def __init__(self):
        super(reg_pos, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, h_pred, h_label):
        l1_loss = h_label[:, 1, :, :] * self.smoothl1(h_pred[:, 0, :, :] /
            (h_label[:, 0, :, :] + 1e-10), h_label[:, 0, :, :] / (h_label[:,
            0, :, :] + 1e-10))
        reg_loss = torch.sum(l1_loss) / max(1.0, torch.sum(h_label[:, 1, :, :])
            )
        return reg_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
