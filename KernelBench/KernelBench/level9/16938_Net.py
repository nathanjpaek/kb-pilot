import torch
import torch.nn.functional as F
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, x_d, w_d, out_d, hidden_d1=256, hidden_d2=512,
        hidden_d3=256, is_discrete_input=False, is_discrete_output=False,
        embedding_dim=None):
        super().__init__()
        self._x_d = x_d
        self._out_d = out_d
        self.is_discrete_input = is_discrete_input
        self.is_discrete_output = is_discrete_output
        if is_discrete_input:
            assert x_d is not None, 'Please specify the dimension of the'
            """treatment vector."""
            embedding_dim = x_d if embedding_dim is None else embedding_dim
            self.embed = nn.Embedding(x_d, embedding_dim)
            in_d = int(embedding_dim + w_d)
        else:
            self.embed = nn.Identity()
            in_d = int(x_d + w_d)
        self.fc1 = nn.Linear(in_d, hidden_d1)
        self.fc2 = nn.Linear(hidden_d1, hidden_d2)
        self.fc3 = nn.Linear(hidden_d2, hidden_d3)
        self.fc4 = nn.Linear(hidden_d3, out_d)

    def forward(self, x, w):
        x = self.embed(x)
        x = torch.cat((x, w), dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        output = self.fc4(x)
        return output


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'x_d': 4, 'w_d': 4, 'out_d': 4}]
