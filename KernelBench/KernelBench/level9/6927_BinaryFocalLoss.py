import torch


class BinaryFocalLoss(torch.nn.Module):
    """ from https://github.com/qubvel/segmentation_models"""

    def __init__(self, gamma=2.0, alpha=0.25, eps=1e-07):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, pr, gt):
        pr = torch.clamp(pr, self.eps, 1 - self.eps)
        loss_1 = -gt * (self.alpha * torch.pow(1 - pr, self.gamma)
            ) * torch.log(pr)
        loss_0 = -(1 - gt) * ((1 - self.alpha) * torch.pow(pr, self.gamma)
            ) * torch.log(1 - pr)
        loss = loss_0 + loss_1
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
