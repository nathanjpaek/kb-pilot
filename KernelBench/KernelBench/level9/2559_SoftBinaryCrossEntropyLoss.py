import torch


class SoftBinaryCrossEntropyLoss(torch.nn.Module):

    def __init__(self, tau=1.0):
        super().__init__()
        self.tau = tau
        self.bce_logit = torch.nn.BCEWithLogitsLoss()

    def forward(self, pred, true):
        logits = pred / self.tau
        l = self.bce_logit(logits, true)
        return l


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
