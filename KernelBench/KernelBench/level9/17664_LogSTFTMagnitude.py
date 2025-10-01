import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class LogSTFTMagnitude(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predicts_mag, targets_mag):
        log_predicts_mag = torch.log(predicts_mag)
        log_targets_mag = torch.log(targets_mag)
        outputs = F.l1_loss(log_predicts_mag, log_targets_mag)
        return outputs


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
