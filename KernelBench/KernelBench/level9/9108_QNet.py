import torch


class QNet(torch.nn.Module):

    def __init__(self, n_features):
        super(QNet, self).__init__()
        self.fc1 = torch.nn.Linear(n_features, 20)
        self.fc1_activate = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc1_activate(x)
        out = self.fc2(x)
        return out

    def init_weights(self):
        for m in self.modules():
            torch.nn.init.normal_(m.weight.data, 0, 0.1)
            torch.nn.init.constant_(m.bias.data, 0.01)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_features': 4}]
