import torch


class BCEFocalLoss(torch.nn.Module):
    """
  二分类的Focalloss alpha 固定
  """

    def __init__(self, gamma=2, alpha=0.25, reduction='sum', loss_weight=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        alpha = self.alpha
        loss = -alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (
            1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss * self.loss_weight / 54


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
