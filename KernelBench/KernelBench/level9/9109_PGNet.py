import torch


class PGNet(torch.nn.Module):

    def __init__(self, n_features, n_actions):
        super(PGNet, self).__init__()
        self.fc1 = torch.nn.Linear(n_features, 20)
        self.fc1_activate = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(20, n_actions)
        self.out_activate = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc1_activate(x)
        x = self.fc2(x)
        out = self.out_activate(x)
        return out

    def init_weights(self):
        for m in self.modules():
            torch.nn.init.normal_(m.weight.data, 0, 0.1)
            torch.nn.init.constant_(m.bias.data, 0.01)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_features': 4, 'n_actions': 4}]
