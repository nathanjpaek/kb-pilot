import torch
from torch import nn


class DiceLoss(nn.Module):

    def __init__(self, epsilon=1e-09):
        """Dice-Loss, 切块损失, 用于不均衡数据, 但是收敛困难, 不太稳定
        paper: Dice Loss for Data-imbalanced NLP Tasks
        url: https://arxiv.org/pdf/1911.02855.pdf
        args:
            reduction: str, Specifies the reduction to apply to the output, 输出形式. 
                            eg.``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``
            epsilon: float, Minimum of maths, 无穷小.  eg. 1e-9
        returns:
            Tensor of loss.
        examples:
            >>> label, logits = [[1, 1, 1, 1], [0, 0, 0, 1]], [[0, 1, 1, 0], [1, 0, 0, 1],]
            >>> label, logits = torch.tensor(label).long(), torch.tensor(logits).float()
            >>> loss = DiceLoss()(logits, label)
        """
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits, labels):
        predict = torch.sigmoid(logits)
        intersect = predict * labels + self.epsilon
        unionset = predict + labels + self.epsilon
        loss = 1 - 2 * intersect.sum() / unionset.sum()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
