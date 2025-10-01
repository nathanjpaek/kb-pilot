import torch
import torch.nn as nn


class BBoxTransform(nn.Module):

    def forward(self, anchors, regression):
        """
        Args:
            anchors: [batch_size, boxes, (y1, x1, y2, x2)]
            regression: [batch_size, boxes, (dy, dx, dh, dw)]
        """
        y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
        x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
        ha = anchors[..., 2] - anchors[..., 0]
        wa = anchors[..., 3] - anchors[..., 1]
        w = regression[..., 3].exp() * wa
        h = regression[..., 2].exp() * ha
        y_centers = regression[..., 0] * ha + y_centers_a
        x_centers = regression[..., 1] * wa + x_centers_a
        ymin = y_centers - h / 2.0
        xmin = x_centers - w / 2.0
        ymax = y_centers + h / 2.0
        xmax = x_centers + w / 2.0
        return torch.stack([xmin, ymin, xmax, ymax], dim=2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
