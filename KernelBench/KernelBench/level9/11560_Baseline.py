import torch
import torch.utils.data
import torch
import torch.nn as nn


class Baseline(nn.Module):
    """Baseline
    """

    def __init__(self, hid_dim, x_dim, binary_dim, inp_dim):
        super(Baseline, self).__init__()
        self.x_dim = x_dim
        self.binary_dim = binary_dim
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.linear1 = nn.Linear(x_dim + self.binary_dim + self.inp_dim,
            self.hid_dim)
        self.linear2 = nn.Linear(self.hid_dim, 1)

    def forward(self, x, binary, inp):
        """Estimate agent's loss based on the agent's input.

        Args:
            x: Image features.
            binary: Communication message.
            inp: Hidden state (used when agent is the Receiver).
        Output:
            score: An estimate of the agent's loss.
        """
        features = []
        if x is not None:
            features.append(x)
        if binary is not None:
            features.append(binary)
        if inp is not None:
            features.append(inp)
        features = torch.cat(features, 1)
        hidden = self.linear1(features).clamp(min=0)
        pred_score = self.linear2(hidden)
        return pred_score


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'hid_dim': 4, 'x_dim': 4, 'binary_dim': 4, 'inp_dim': 4}]
