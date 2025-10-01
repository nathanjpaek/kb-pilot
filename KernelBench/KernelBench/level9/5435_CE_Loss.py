import torch
from torch import nn
import torch.nn.functional as F
import torch.utils


class CE_Loss(nn.Module):

    def __init__(self, temperature=1):
        super(CE_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1)
        loss = -self.T * self.T * torch.sum(torch.mul(output_batch,
            teacher_outputs)) / teacher_outputs.size(0)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
