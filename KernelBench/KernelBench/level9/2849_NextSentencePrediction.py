import torch
import torch.nn as nn
import torch.cuda
import torch.distributed


class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_random_next

    Args:
            hidden_size (int): BERT model output size
    """

    def __init__(self, hidden_size):
        super(NextSentencePrediction, self).__init__()
        self.linear = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x (Tensor): second output of bert encoder, ``(B, H)``
        Returns:
            seq_class_prob (Tensor): ``(B, 2)``
        """
        seq_relationship_score = self.linear(x)
        seq_class_log_prob = self.log_softmax(seq_relationship_score)
        return seq_class_log_prob


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
