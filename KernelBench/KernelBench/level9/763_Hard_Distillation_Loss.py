import torch
import torch.nn as nn
import torch.nn


class Hard_Distillation_Loss(nn.Module):

    def __init__(self):
        super(Hard_Distillation_Loss, self).__init__()
        self.CE_teacher = nn.CrossEntropyLoss()
        self.CE_student = nn.CrossEntropyLoss()

    def forward(self, teacher_y, student_y, y):
        loss = 1 / 2 * self.CE_student(student_y, y) + 1 / 2 * self.CE_teacher(
            teacher_y, y)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
