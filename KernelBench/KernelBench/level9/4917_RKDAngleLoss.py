import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwaise_distance(output):
    """
    Function for calculating pairwise distance

    :param output (torch.FloatTensor): Input for calculating pairwise distance
    """
    output_squared = output.pow(2).sum(dim=1)
    product = torch.mm(output, output.t())
    result = output_squared.unsqueeze(0) + output_squared.unsqueeze(1
        ) - 2 * product
    result[range(len(output)), range(len(output))] = 0
    return result.sqrt()


class RKDAngleLoss(nn.Module):
    """
    Module for calculating RKD Angle Loss
    """

    def forward(self, teacher, student, normalize=False):
        """
        Forward function

        :param teacher (torch.FloatTensor): Prediction made by the teacher model
        :param student (torch.FloatTensor): Prediction made by the student model 
        :param normalize (bool): True if inputs need to be normalized 
        """
        with torch.no_grad():
            t = pairwaise_distance(teacher)
            if normalize:
                t = F.normalize(t, p=2, dim=2)
        s = pairwaise_distance(student)
        if normalize:
            s = F.normalize(s, p=2, dim=2)
        return F.smooth_l1_loss(s, t, reduction='mean')


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
