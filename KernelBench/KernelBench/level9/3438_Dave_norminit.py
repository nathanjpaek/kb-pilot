import torch
import torch.nn as nn
import torch.utils.data


class Dave_norminit(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, (5, 5), stride=(2, 2))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(24, 36, (5, 5), stride=(2, 2))
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(36, 48, (5, 5), stride=(2, 2))
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(48, 64, (3, 3), stride=(1, 1))
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(64, 64, (3, 3), stride=(1, 1))
        self.relu5 = nn.ReLU()
        self.fc1 = nn.Linear(1600, 1164)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(1164, 100)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.1)
        self.relu7 = nn.ReLU()
        self.fc3 = nn.Linear(100, 50)
        nn.init.normal_(self.fc3.weight, mean=0.0, std=0.1)
        self.relu8 = nn.ReLU()
        self.fc4 = nn.Linear(50, 10)
        nn.init.normal_(self.fc4.weight, mean=0.0, std=0.1)
        self.relu9 = nn.ReLU()
        self.before_prediction = nn.Linear(10, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = x.view(-1, 1600)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.fc2(x)
        x = self.relu7(x)
        x = self.fc3(x)
        x = self.relu8(x)
        x = self.fc4(x)
        x = self.relu9(x)
        x = self.before_prediction(x)
        x = torch.atan(x) * 2
        return x


def get_inputs():
    return [torch.rand([4, 3, 96, 96])]


def get_init_inputs():
    return [[], {}]
