import torch


def square(x):
    return x * x


class Net1(torch.nn.Module):

    def __init__(self, hidden=64, output=10):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
        self.fc1 = torch.nn.Linear(256, hidden)
        self.fc2 = torch.nn.Linear(hidden, output)

    def forward(self, x):
        x = self.conv1(x)
        x = square(x)
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = square(x)
        x = self.fc2(x)
        return x

    def mid_layer(self):
        return ['o1', 'o1a', 'o2', 'o2a', 'o3']

    def forward_analyze(self, x):
        o1 = self.conv1(x)
        o1a = square(o1)
        o1a = x.view(-1, 256)
        o2 = self.fc1(o1a)
        o2a = square(o2)
        o3 = self.fc2(o2a)
        return o1, o1a, o2, o2a, o3


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
