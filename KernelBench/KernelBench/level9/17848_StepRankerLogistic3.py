import torch
from torch import nn


class StepRankerLogistic3(nn.Module):
    """a logistic ranker that includes a don't care token"""

    def __init__(self, parent_dim, child_short_dim, child_full_dim, hidden_dim
        ):
        super(StepRankerLogistic3, self).__init__()
        if child_full_dim is not None:
            self.hidden = nn.Linear(parent_dim + child_short_dim +
                child_full_dim + child_short_dim + child_full_dim, hidden_dim)
        else:
            self.hidden = nn.Linear(parent_dim + child_short_dim +
                child_short_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, 3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, parent_encodings, child_short_encodings_1,
        child_short_encodings_2, child_full_encodings_1, child_full_encodings_2
        ):
        """return the predicted probability that child 1 should come temporally before child 2"""
        assert (child_full_encodings_1 is None) == (child_full_encodings_2 is
            None)
        if child_full_encodings_1 is not None:
            input = torch.cat([parent_encodings, child_short_encodings_1,
                child_full_encodings_1, child_short_encodings_2,
                child_full_encodings_2], dim=1)
        else:
            input = torch.cat([parent_encodings, child_short_encodings_1,
                child_short_encodings_2], dim=1)
        return self.logsoftmax(self.out(self.relu(self.hidden(input))))


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]),
        torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'parent_dim': 4, 'child_short_dim': 4, 'child_full_dim': 4,
        'hidden_dim': 4}]
