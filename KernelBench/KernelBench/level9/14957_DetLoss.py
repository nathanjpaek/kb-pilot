import torch
from torch import nn


class DetLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.hm_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.ori_criterion = nn.SmoothL1Loss(reduction='none')
        self.box_criterion = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred_heatmaps, heatmaps, pred_sizemaps, sizemaps,
        pred_orimaps, orimaps):
        size_w, _ = heatmaps.max(dim=1, keepdim=True)
        p_det = torch.sigmoid(pred_heatmaps * (1 - 2 * heatmaps))
        det_loss = (self.hm_criterion(pred_heatmaps, heatmaps) * p_det).mean(
            ) / p_det.mean()
        box_loss = (size_w * self.box_criterion(pred_sizemaps, sizemaps)).mean(
            ) / size_w.mean()
        ori_loss = (size_w * self.ori_criterion(pred_orimaps, orimaps)).mean(
            ) / size_w.mean()
        return det_loss, box_loss, ori_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]),
        torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
