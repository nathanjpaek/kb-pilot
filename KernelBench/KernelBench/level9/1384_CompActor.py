import torch


class CompActor(torch.nn.Module):

    def __init__(self, state_dim: 'int', hidden_dim: 'int', action_dim: 'int'):
        super(CompActor, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        self.fc4 = torch.nn.Linear(action_dim, action_dim)
        self.fc5 = torch.nn.Linear(action_dim, action_dim)
        torch.nn.init.constant_(self.fc4.weight, 0)
        torch.nn.init.constant_(self.fc4.bias, 0)
        torch.nn.init.constant_(self.fc5.weight, 0)
        torch.nn.init.constant_(self.fc5.bias, -1)
        with torch.no_grad():
            for idx, elem in enumerate(self.fc4.weight):
                elem[idx] = 2
            for idx, elem in enumerate(self.fc4.weight):
                elem[idx] = 2

    def forward(self, x):
        z = torch.relu(self.fc1(x))
        z = torch.relu(self.fc2(z))
        z = self.fc3(z)
        z = torch.sigmoid(self.fc4(z))
        z = self.fc5(z)
        return z


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'hidden_dim': 4, 'action_dim': 4}]
