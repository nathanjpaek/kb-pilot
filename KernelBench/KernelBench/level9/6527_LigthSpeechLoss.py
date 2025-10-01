import torch
from torch import nn
import torch.utils.data


class LigthSpeechLoss(nn.Module):
    """ LigthSpeech Loss """

    def __init__(self):
        super(LigthSpeechLoss, self).__init__()

    def forward(self, mel, padd_predicted, cemb_out, mel_tac2_target, D, cemb):
        mel_loss = nn.MSELoss()(mel, mel_tac2_target)
        similarity_loss = nn.L1Loss()(cemb_out, cemb)
        duration_loss = nn.L1Loss()(padd_predicted, D.float())
        return mel_loss, similarity_loss, duration_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]),
        torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
