import torch
import torch.nn as nn
import torch.utils.data


class IoULoss(nn.Module):
    """Some Information about IoULoss"""

    def forward(self, preds: 'torch.Tensor', targets: 'torch.Tensor', eps:
        'float'=1e-08) ->torch.Tensor:
        """IoU Loss

        Args:
            preds (torch.Tensor): [x1, y1, x2, y2] predictions [*, 4]
            targets (torch.Tensor): [x1, y1, x2, y2] targets [*, 4]

        Returns:
            torch.Tensor: [-log(iou)] [*]
        """
        lt = torch.max(preds[..., :2], targets[..., :2])
        rb = torch.min(preds[..., 2:], targets[..., 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]
        ap = (preds[..., 2] - preds[..., 0]) * (preds[..., 3] - preds[..., 1])
        ag = (targets[..., 2] - targets[..., 0]) * (targets[..., 3] -
            targets[..., 1])
        union = ap + ag - overlap + eps
        ious = overlap / union
        ious = torch.clamp(ious, min=eps)
        return -ious.log()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
