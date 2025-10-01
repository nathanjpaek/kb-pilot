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


class Accuracy(nn.Module):

    def __init__(self, bback_ignore=True):
        super(Accuracy, self).__init__()
        self.bback_ignore = bback_ignore

    def forward(self, y_pred, y_true):
        _n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        prob = F.softmax(y_pred, dim=1).data
        prediction = torch.argmax(prob, 1)
        accs = []
        for c in range(int(self.bback_ignore), ch):
            yt_c = y_true[:, c, ...]
            num = (prediction.eq(c) + yt_c.data.eq(1)).eq(2).float().sum() + 1
            den = yt_c.data.eq(1).float().sum() + 1
            acc = num / den * 100
            accs.append(acc)
        accs = torch.stack(accs)
        return accs.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
