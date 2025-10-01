import torch
import torch.utils.data
from torch import nn


class Tacotron2Loss(nn.Module):

    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_out_before, mel_out_after, gate_out, _ = model_output
        mel_loss = nn.MSELoss()(mel_out_before, mel_target) + nn.MSELoss()(
            mel_out_after, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out.view(-1, 1),
            gate_target.view(-1, 1))
        return mel_loss + gate_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
