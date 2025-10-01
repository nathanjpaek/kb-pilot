import torch
from torch import nn
import torch.nn.functional as F


class DistilledLoss(nn.Module):
    """
    Intended for use with a DistillationTrainer.
    Combines vanilla cross entropy loss with a modified form of KL divergence loss.
    Attempts to minimize the KL divergence between the student and distilled logits
    while maintaining an emphasis on predicting the true labels with cross entropy.
    """

    def __init__(self, alpha=0.5, temperature=1.0):
        super(DistilledLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, teacher_logits, student_logits, distill_logits, labels):
        loss = F.cross_entropy(student_logits, labels)
        distill_loss = F.kl_div(F.log_softmax(distill_logits / self.
            temperature, dim=-1), F.softmax(teacher_logits / self.
            temperature, dim=-1).detach(), reduction='batchmean')
        distill_loss *= self.temperature ** 2
        return loss * self.alpha + distill_loss * (1 - self.alpha)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
