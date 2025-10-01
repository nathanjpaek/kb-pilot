import torch


class Network(torch.nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(64, 512, kernel_size=5)
        self.fc1 = torch.nn.Linear(2048, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=3)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=2)
        x = x.view(-1, 2048)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        return torch.nn.functional.log_softmax(x, dim=1)


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
