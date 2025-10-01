import torch


def square(x):
    return x * x


class Net2(torch.nn.Module):

    def __init__(self, act=square, output=10):
        super().__init__()
        self.act = act
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(8, 64, kernel_size=3, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=0)
        self.fc = torch.nn.Linear(4 * 3 * 3, output)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.act(x)
        x = x.view(-1, 36)
        x = self.fc(x)
        return x

    def mid_layer(self):
        return ['o1', 'o1a', 'o2', 'o2a', 'o3', 'o3a', 'o4']

    def forward_analyze(self, x):
        o1 = self.conv1(x)
        o1a = self.act(o1)
        o2 = self.conv2(o1a)
        o2a = self.act(o2)
        o3 = self.conv3(o2a)
        o3a = self.act(o3)
        o3a = x.view(-1, 36)
        o4 = self.fc(o3a)
        return o1, o1a, o2, o2a, o3, o3a, o4


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
