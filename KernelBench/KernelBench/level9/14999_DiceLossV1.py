import torch
from torch import nn


class DiceLossV1(nn.Module):

    def __init__(self, reduction='mean', epsilon=1e-09):
        """【ERROR, 不收敛-原因未知】Dice-Loss, 切块损失, 用于不均衡数据, 但是收敛困难
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
            >>> label, logits = torch.tensor(label).float(), torch.tensor(logits).float()
            >>> loss = DiceLoss()(logits, label)
        """
        super(DiceLossV1, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, logits, labels):
        prob = torch.sigmoid(logits)
        index = labels.unsqueeze(1).view(prob.size(0), -1)
        prob = torch.gather(prob, dim=1, index=index)
        dsc_i = 1 - ((1 - prob) * prob + self.epsilon) / ((1 - prob) * prob +
            1 + self.epsilon)
        if 'mean' == self.reduction:
            loss = dsc_i.mean()
        else:
            loss = dsc_i.sum()
        return loss


def get_inputs():
    return [torch.ones([4, 4], dtype=torch.int64), torch.ones([4], dtype=
        torch.int64)]


def get_init_inputs():
    return [[], {}]
