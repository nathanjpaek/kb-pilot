import torch
import torch.utils.cpp_extension


class SoftmaxLoss(torch.nn.Module):

    def __init__(self, tau=1.0):
        super().__init__()
        self.tau = tau
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, pred, true):
        logits = pred / self.tau
        l = self.ce_loss(logits, true)
        return l


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
