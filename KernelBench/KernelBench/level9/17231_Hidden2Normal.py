import torch


class Hidden2Normal(torch.nn.Module):

    def __init__(self, hidden_dim):
        super(Hidden2Normal, self).__init__()
        self.linear = torch.nn.Linear(hidden_dim, 5)

    def forward(self, hidden_state):
        normal = self.linear(hidden_state)
        normal[:, 2] = 0.01 + 0.2 * torch.sigmoid(normal[:, 2])
        normal[:, 3] = 0.01 + 0.2 * torch.sigmoid(normal[:, 3])
        normal[:, 4] = 0.7 * torch.sigmoid(normal[:, 4])
        return normal


def get_inputs():
    return [torch.rand([4, 5, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_dim': 4}]
