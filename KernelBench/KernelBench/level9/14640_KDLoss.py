import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):

    def __init__(self, temp: 'float', reduction: 'str'):
        super(KDLoss, self).__init__()
        self.temp = temp
        self.reduction = reduction
        self.kl_loss = nn.KLDivLoss(reduction=reduction)

    def forward(self, teacher_logits: 'torch.Tensor', student_logits:
        'torch.Tensor'):
        student_softmax = F.log_softmax(student_logits / self.temp, dim=-1)
        teacher_softmax = F.softmax(teacher_logits / self.temp, dim=-1)
        kl = nn.KLDivLoss(reduction='none')(student_softmax, teacher_softmax)
        kl = kl.sum() if self.reduction == 'sum' else kl.sum(1).mean()
        kl = kl * self.temp ** 2
        return kl

    def __call__(self, *args, **kwargs):
        return super(KDLoss, self).__call__(*args, **kwargs)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'temp': 4, 'reduction': 4}]
