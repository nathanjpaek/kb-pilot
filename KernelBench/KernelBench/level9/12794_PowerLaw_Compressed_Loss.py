import torch
import torch.nn as nn
import torch.utils.data


class PowerLaw_Compressed_Loss(nn.Module):

    def __init__(self, power=0.3, complex_loss_ratio=0.113):
        super(PowerLaw_Compressed_Loss, self).__init__()
        self.power = power
        self.complex_loss_ratio = complex_loss_ratio
        self.criterion = nn.MSELoss()
        self.epsilon = 1e-16

    def forward(self, prediction, target, seq_len=None, spec_phase=None):
        prediction = prediction + self.epsilon
        target = target + self.epsilon
        prediction = torch.pow(prediction, self.power)
        target = torch.pow(target, self.power)
        spec_loss = self.criterion(torch.abs(target), torch.abs(prediction))
        complex_loss = self.criterion(target, prediction)
        loss = spec_loss + complex_loss * self.complex_loss_ratio
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
