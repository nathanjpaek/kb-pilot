import torch
import torch.nn as nn


class MultiSampleDropout(nn.Module):
    """
    # multisample dropout (wut): https://arxiv.org/abs/1905.09788
    """

    def __init__(self, hidden_size, num_labels, K=5, p=0.5):
        super().__init__()
        self.K = K
        self.dropout = nn.Dropout(p)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input):
        logits = torch.stack([self.classifier(self.dropout(input)) for _ in
            range(self.K)], dim=0)
        logits = torch.mean(logits, dim=0)
        return logits


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'num_labels': 4}]
