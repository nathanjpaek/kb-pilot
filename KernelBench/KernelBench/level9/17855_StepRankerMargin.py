import torch
from torch import nn


class StepRankerMargin(nn.Module):

    def __init__(self, parent_dim, child_short_dim, child_full_dim, hidden_dim
        ):
        super(StepRankerMargin, self).__init__()
        if child_full_dim is not None:
            self.hidden = nn.Linear(parent_dim + child_short_dim +
                child_full_dim, hidden_dim)
        else:
            self.hidden = nn.Linear(parent_dim + child_short_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, parent_encodings, child_short_encodings_1,
        child_short_encodings_2, child_full_encodings_1, child_full_encodings_2
        ):
        """return the ranking scores for child 1 and child 2 in which child 1 should come temporally before child 2"""
        assert (child_full_encodings_1 is None) == (child_full_encodings_2 is
            None)
        if child_full_encodings_1 is not None:
            encodings_1 = torch.cat([parent_encodings,
                child_short_encodings_1, child_full_encodings_1], dim=1)
            encodings_2 = torch.cat([parent_encodings,
                child_short_encodings_2, child_full_encodings_2], dim=1)
        else:
            encodings_1 = torch.cat([parent_encodings,
                child_short_encodings_1], dim=1)
            encodings_2 = torch.cat([parent_encodings,
                child_short_encodings_2], dim=1)
        score_1 = self.out(self.relu(self.hidden(encodings_1)))
        score_2 = self.out(self.relu(self.hidden(encodings_2)))
        return score_1.view(score_1.numel()), score_2.view(score_2.numel())

    def score(self, parent_encodings, child_encodings):
        """return the score of multiple child encodings each with respective parent encoding"""
        encodings = torch.cat([parent_encodings, child_encodings], dim=1)
        scores = self.out(self.relu(self.hidden(encodings)))
        return scores.view(scores.numel())


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]),
        torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'parent_dim': 4, 'child_short_dim': 4, 'child_full_dim': 4,
        'hidden_dim': 4}]
