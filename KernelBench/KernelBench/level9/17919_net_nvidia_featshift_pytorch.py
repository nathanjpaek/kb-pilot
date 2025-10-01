import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class net_nvidia_featshift_pytorch(nn.Module):

    def __init__(self):
        super(net_nvidia_featshift_pytorch, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, 5, 2)
        self.conv2 = nn.Conv2d(24, 36, 5, 2)
        self.conv3 = nn.Conv2d(36, 48, 5, 2)
        self.conv4 = nn.Conv2d(48, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 1 * 18, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, img, feature, mean2, std2):
        x = LambdaLayer(lambda x: x / 127.5 - 1.0)(img)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = self.conv5(x)
        mean1 = torch.mean(x)
        std1 = torch.std(x)
        f = torch.sub(feature, mean2)
        f = torch.div(f, std2)
        f = torch.mul(f, std1)
        f = torch.add(f, mean1)
        x = F.elu(x)
        x = x.view(-1, 64 * 1 * 18)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        f = F.elu(f)
        f = f.view(-1, 64 * 1 * 18)
        f = F.elu(self.fc1(f))
        f = F.elu(self.fc2(f))
        f = F.elu(self.fc3(f))
        f = self.fc4(f)
        x = torch.cat((x, f), 0)
        return x


def get_inputs():
    return [torch.rand([4, 3, 128, 128]), torch.rand([4, 1152]), torch.rand
        ([4, 4, 4, 1152]), torch.rand([4, 4, 4, 1152])]


def get_init_inputs():
    return [[], {}]
