import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data


class InteractiveKLLoss(nn.Module):

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss()

    def forward(self, student, teacher):
        return self.kl_loss(F.log_softmax(student / self.temperature, dim=1
            ), F.softmax(teacher / self.temperature, dim=1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'temperature': 4}]
