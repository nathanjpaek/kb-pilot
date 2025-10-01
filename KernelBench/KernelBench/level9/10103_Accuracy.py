import torch
import torch.nn.functional as F
import torch.nn as nn


class Accuracy(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, prediction, target, mask=None, token_dim=-1,
        sequence_dim=-2):
        prediction = F.softmax(prediction, token_dim).argmax(sequence_dim)
        scores = prediction == target
        n_padded = 0
        if mask is not None:
            n_padded = (mask == 0).sum()
        return scores.sum() / float(scores.numel() - n_padded)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
