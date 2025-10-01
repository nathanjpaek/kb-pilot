import torch


class ActionApproximation(torch.nn.Module):

    def __init__(self, state_observations_count, action_count, hidden_count=512
        ):
        super(ActionApproximation, self).__init__()
        self.ReLU = torch.nn.ReLU()
        self.dense0 = torch.nn.Linear(state_observations_count, hidden_count)
        self.dense1 = torch.nn.Linear(hidden_count, hidden_count)
        self.dense2 = torch.nn.Linear(hidden_count, action_count)

    def forward(self, x):
        x = x.float()
        x = self.dense0(x)
        x = self.ReLU(x)
        x = self.dense1(x)
        x = self.ReLU(x)
        x = self.dense2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_observations_count': 4, 'action_count': 4}]
