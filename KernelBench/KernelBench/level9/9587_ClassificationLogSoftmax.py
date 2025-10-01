import torch
import torch.nn as nn


class ClassificationLogSoftmax(nn.Module):
    """
    Classifier on top of the hidden representation of the first token, which
    is usually [CLS] token in BERT-like architectures.
    """

    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states):
        output_states = self.dense1(hidden_states[:, 0])
        output_states = torch.tanh(output_states)
        output_states = self.dense2(output_states).float()
        log_probs = torch.log_softmax(output_states, dim=-1)
        return log_probs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'num_classes': 4}]
