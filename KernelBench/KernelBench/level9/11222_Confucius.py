import torch
import torch.nn as nn


class Confucius(nn.Module):

    def __init__(self, output_dim, expose_dim, hidden):
        super(Confucius, self).__init__()
        self.output_fc = nn.Linear(output_dim, hidden)
        self.fc_expose = nn.Linear(expose_dim, hidden)
        self.fc_final = nn.Linear(hidden, 1)

    def forward(self, output, expose):
        out1 = self.output_fc(output)
        out2 = self.fc_expose(expose)
        out = self.fc_final(out1 + out2)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'output_dim': 4, 'expose_dim': 4, 'hidden': 4}]
