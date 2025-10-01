import torch
import torch.nn


class SkipModule(torch.nn.Module):

    def __init__(self, in_features, out_features, activation=torch.nn.ReLU()):
        super(SkipModule, self).__init__()
        self.linear1 = torch.nn.Linear(in_features, out_features, activation)
        self.linear2 = torch.nn.Linear(out_features, out_features, activation)
        self.linear3 = torch.nn.Linear(in_features + out_features,
            out_features, activation)

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.linear2(x1)
        x = torch.cat((x, x1), dim=-1)
        return self.linear3(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
