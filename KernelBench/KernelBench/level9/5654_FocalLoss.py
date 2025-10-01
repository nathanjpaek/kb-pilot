import torch


class FocalLoss(torch.nn.Module):

    def __init__(self, gamma=2, alpha=0.5, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 1e-06

    def forward(self, logits, labels):
        pt = torch.sigmoid(logits)
        alpha = self.alpha
        loss = -alpha * (1 - pt) ** self.gamma * labels * torch.log(pt) - (
            1 - alpha) * pt ** self.gamma * (1 - labels) * torch.log(1 - pt)
        return torch.mean(loss)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
