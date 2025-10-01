import torch
import torch.nn as nn


class iRMSE(nn.Module):

    def __init__(self):
        super(iRMSE, self).__init__()

    def forward(self, outputs, target, *args):
        outputs = outputs / 1000.0
        target = target / 1000.0
        outputs[outputs == 0] = -1
        target[target == 0] = -1
        outputs = 1.0 / outputs
        target = 1.0 / target
        outputs[outputs == -1] = 0
        target[target == -1] = 0
        val_pixels = (target > 0).float()
        err = (target * val_pixels - outputs * val_pixels) ** 2
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1,
            keepdim=True)
        return torch.mean(torch.sqrt(loss / cnt))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
