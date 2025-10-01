import torch
import torch.nn.functional as F
import torch.nn as nn


class LabelwiseAttention(nn.Module):
    """Applies attention technique to summarize the sequence for each label
    See `Explainable Prediction of Medical Codes from Clinical Text <https://aclanthology.org/N18-1100.pdf>`_

    Args:
        input_size (int): The number of expected features in the input.
        num_classes (int): Total number of classes.
    """

    def __init__(self, input_size, num_classes):
        super(LabelwiseAttention, self).__init__()
        self.attention = nn.Linear(input_size, num_classes, bias=False)

    def forward(self, input):
        attention = self.attention(input).transpose(1, 2)
        attention = F.softmax(attention, -1)
        logits = torch.bmm(attention, input)
        return logits, attention


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'num_classes': 4}]
