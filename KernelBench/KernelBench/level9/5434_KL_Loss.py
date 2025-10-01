import torch
from torch import nn
import torch.nn.functional as F
import torch.utils


class KL_Loss(nn.Module):

    def __init__(self, temperature=1):
        super(KL_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** -7
        loss = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(
            output_batch, teacher_outputs)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
