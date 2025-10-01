import torch
import torch.nn as nn
import torch.nn


class Soft_Distillation_Loss(nn.Module):

    def __init__(self, lambda_balancing):
        super(Soft_Distillation_Loss, self).__init__()
        self.lambda_balancing = lambda_balancing
        self.CE_student = nn.CrossEntropyLoss()
        self.KLD_teacher = nn.KLDivLoss()

    def forward(self, teacher_y, student_y, y, temperature):
        loss = (1 - self.lambda_balancing) * self.CE_student(student_y, y
            ) + self.lambda_balancing * temperature ** 2 * self.KLD_teacher(
            student_y / temperature, teacher_y / temperature)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'lambda_balancing': 4}]
