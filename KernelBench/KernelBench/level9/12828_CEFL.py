import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx


class CEFL(nn.Module):

    def __init__(self, gamma=1):
        super(CEFL, self).__init__()
        self.gamma = gamma

    def get_prob(self, input, target):
        prob = F.softmax(input, dim=-1)
        prob = prob[range(target.shape[0]), target]
        return prob

    def get_attention(self, input, target):
        prob = self.get_prob(input, target)
        prob = 1 - prob
        prob = prob ** self.gamma
        return prob

    def get_celoss(self, input, target):
        ce_loss = F.log_softmax(input, dim=1)
        ce_loss = -ce_loss[range(target.shape[0]), target]
        return ce_loss

    def forward(self, input, target):
        attn = self.get_attention(input, target)
        ce_loss = self.get_celoss(input, target)
        prob = self.get_prob(input, target)
        loss = (1 - prob) * ce_loss + prob * attn * ce_loss
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {}]
