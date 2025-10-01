import torch
import torch.nn.functional
import torch.nn as nn
import torch.nn.functional as F


def centercrop(image, w, h):
    _nt, _ct, ht, wt = image.size()
    padw, padh = (wt - w) // 2, (ht - h) // 2
    if padw > 0 and padh > 0:
        image = image[:, :, padh:-padh, padw:-padw]
    return image


def flatten(x):
    x_flat = x.clone()
    x_flat = x_flat.view(x.shape[0], -1)
    return x_flat


class Dice(nn.Module):

    def __init__(self, bback_ignore=True):
        super(Dice, self).__init__()
        self.bback_ignore = bback_ignore

    def forward(self, y_pred, y_true):
        eps = 1e-15
        _n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        prob = F.softmax(y_pred, dim=1)
        prob = prob.data
        prediction = torch.argmax(prob, dim=1)
        y_pred_f = flatten(prediction).float()
        dices = []
        for c in range(int(self.bback_ignore), ch):
            y_true_f = flatten(y_true[:, c, ...]).float()
            intersection = y_true_f * y_pred_f
            dice = 2.0 * torch.sum(intersection) / (torch.sum(y_true_f) +
                torch.sum(y_pred_f) + eps) * 100
            dices.append(dice)
        dices = torch.stack(dices)
        return dices.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
