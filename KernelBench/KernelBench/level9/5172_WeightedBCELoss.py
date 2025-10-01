import torch
import torch.nn.functional
import torch.nn as nn


def centercrop(image, w, h):
    _nt, _ct, ht, wt = image.size()
    padw, padh = (wt - w) // 2, (ht - h) // 2
    if padw > 0 and padh > 0:
        image = image[:, :, padh:-padh, padw:-padw]
    return image


class WeightedBCELoss(nn.Module):

    def __init__(self):
        super(WeightedBCELoss, self).__init__()

    def forward(self, y_pred, y_true, weight):
        _n, _ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        weight = centercrop(weight, w, h)
        logit_y_pred = torch.log(y_pred / (1.0 - y_pred))
        loss = weight * (logit_y_pred * (1.0 - y_true) + torch.log(1.0 +
            torch.exp(-torch.abs(logit_y_pred))) + torch.clamp(-
            logit_y_pred, min=0.0))
        loss = torch.sum(loss) / torch.sum(weight)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
