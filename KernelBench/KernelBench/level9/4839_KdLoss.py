import torch
import torch.nn.functional as F
import torch.utils
import torch.utils.data.distributed


class KdLoss(torch.nn.Module):

    def __init__(self, alpha=0.9, T=5):
        super(KdLoss, self).__init__()
        self.alpha = alpha
        self.T = T
        self.criterion = torch.nn.KLDivLoss()

    def forward(self, outputs, teacher_outputs, labels):
        alpha = self.alpha
        T = self.T
        KD_loss = self.criterion(F.log_softmax(outputs / T, dim=1), F.
            softmax(teacher_outputs / T, dim=1)) * (alpha * T * T
            ) + F.cross_entropy(outputs, labels) * (1.0 - alpha)
        return KD_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
