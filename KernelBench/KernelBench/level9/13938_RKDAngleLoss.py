import torch
import torch.nn as nn
from torch.nn import functional as F


class RKDAngleLoss(nn.Module):
    """
    Module for calculating RKD Angle Loss
    """

    def forward(self, teacher, student, normalize=True):
        """
        Forward function

        :param teacher (torch.FloatTensor): Prediction made by the teacher model
        :param student (torch.FloatTensor): Prediction made by the student model
        :param normalize (bool): True if inputs need to be normalized
        """
        with torch.no_grad():
            t = teacher.unsqueeze(0) - teacher.unsqueeze(1)
            if normalize:
                t = F.normalize(t, p=2, dim=2)
            t = torch.bmm(t, t.transpose(1, 2)).view(-1)
        s = student.unsqueeze(0) - student.unsqueeze(1)
        if normalize:
            s = F.normalize(s, p=2, dim=2)
        s = torch.bmm(s, s.transpose(1, 2)).view(-1)
        return F.smooth_l1_loss(s, t)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
