import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionTransfer(nn.Module):

    def forward(self, student, teacher):
        s_attention = F.normalize(student.pow(2).mean(1).view(student.size(
            0), -1))
        with torch.no_grad():
            t_attention = F.normalize(teacher.pow(2).mean(1).view(teacher.
                size(0), -1))
        return (s_attention - t_attention).pow(2).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
