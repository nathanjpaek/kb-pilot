import torch
import torch.nn.functional as F
import torch.utils
import torch.utils.data.distributed
from torch.nn.modules import loss


class DistributionLoss(loss._Loss):

    def forward(self, model_output, real_output):
        self.size_average = True
        if real_output.requires_grad:
            raise ValueError(
                'real network output should not require gradients.')
        model_output_log_prob = F.log_softmax(model_output, dim=1)
        real_output_soft = F.softmax(real_output, dim=1)
        del model_output, real_output
        real_output_soft = real_output_soft.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(real_output_soft, model_output_log_prob
            )
        if self.size_average:
            cross_entropy_loss = cross_entropy_loss.mean()
        else:
            cross_entropy_loss = cross_entropy_loss.sum()
        return cross_entropy_loss


class DistillationLoss(torch.nn.Module):

    def __init__(self, alpha=0.9):
        super(DistillationLoss, self).__init__()
        self.criterion1 = torch.nn.CrossEntropyLoss()
        self.criterion2 = DistributionLoss()
        self.alpha = alpha

    def forward(self, stu_model_output, tea_model_output, target):
        loss1 = self.criterion1(stu_model_output, target)
        loss2 = self.criterion2(stu_model_output, tea_model_output)
        loss = self.alpha * loss2 + (1.0 - self.alpha) * loss1
        return loss, loss1


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
