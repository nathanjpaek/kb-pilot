import torch
import torch.nn as nn


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class LSTMClassCriterion(nn.Module):

    def __init__(self):
        super(LSTMClassCriterion, self).__init__()

    def forward(self, pred, target, mask):
        pred = pred.clone()
        target = target.clone()
        mask = mask.clone()
        target = target[:, :pred.size(1)]
        mask = mask[:, :pred.size(1)]
        pred = to_contiguous(pred).view(-1, pred.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        loss = -pred.gather(1, target) * mask
        loss = torch.sum(loss) / torch.sum(mask)
        _, idx = torch.max(pred, dim=1)
        correct = idx.eq(torch.squeeze(target))
        correct = correct.float() * torch.squeeze(mask)
        accuracy = torch.sum(correct) / torch.sum(mask)
        return loss, accuracy


def get_inputs():
    return [torch.ones([4, 4, 4], dtype=torch.int64), torch.ones([4, 4],
        dtype=torch.int64), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
