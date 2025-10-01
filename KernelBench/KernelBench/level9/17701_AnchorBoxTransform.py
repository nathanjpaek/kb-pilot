import torch
from torch import Tensor
from typing import Optional
import torch.nn as nn


class AnchorBoxTransform(nn.Module):

    def __init__(self, mean: 'Optional[Tensor]'=None, std:
        'Optional[Tensor]'=None, log_length: 'bool'=False):
        super(AnchorBoxTransform, self).__init__()
        self.mean = mean
        self.std = std
        self.log_length = log_length

    def forward(self, boxes: 'Tensor', deltas: 'Tensor') ->Tensor:
        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        center_x = boxes[:, :, 0] + 0.5 * widths
        center_y = boxes[:, :, 1] + 0.5 * heights
        if self.std is not None:
            deltas = deltas.mul(self.std)
        if self.mean is not None:
            deltas = deltas.add(self.mean)
        dx, dy, dw, dh = [deltas[:, :, i] for i in range(4)]
        if self.log_length:
            dw, dh = [torch.exp(x) for x in (dw, dh)]
        pred_center_x = center_x + dx * widths
        pred_center_y = center_y + dy * heights
        pred_w = dw * widths
        pred_h = dh * heights
        pred_boxes_x1 = pred_center_x - 0.5 * pred_w
        pred_boxes_y1 = pred_center_y - 0.5 * pred_h
        pred_boxes_x2 = pred_center_x + 0.5 * pred_w
        pred_boxes_y2 = pred_center_y + 0.5 * pred_h
        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1,
            pred_boxes_x2, pred_boxes_y2], dim=-1)
        return pred_boxes


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
