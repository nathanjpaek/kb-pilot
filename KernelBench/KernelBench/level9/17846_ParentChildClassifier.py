import torch
from torch import nn


class ParentChildClassifier(nn.Module):

    def __init__(self, parent_dim, child_short_dim, child_full_dim, hidden_dim
        ):
        super(ParentChildClassifier, self).__init__()
        if child_full_dim is not None:
            self.hidden = nn.Linear(parent_dim + child_short_dim +
                child_full_dim, hidden_dim)
        else:
            self.hidden = nn.Linear(parent_dim + child_short_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, parent_encodings, child_short_encodings,
        child_full_encodings):
        if child_full_encodings is not None:
            encodings = torch.cat([parent_encodings, child_short_encodings,
                child_full_encodings], dim=1)
        else:
            encodings = torch.cat([parent_encodings, child_short_encodings],
                dim=1)
        log_probs = self.logsoftmax(self.out(self.relu(self.hidden(encodings)))
            )
        return log_probs


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'parent_dim': 4, 'child_short_dim': 4, 'child_full_dim': 4,
        'hidden_dim': 4}]
