import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class Net(nn.Module):

    def __init__(self, num_rej=0):
        super(Net, self).__init__()
        self.num_rej = num_rej + 1
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10 + self.num_rej)
        self.loss = lambda g: -F.logsigmoid(-g)

    def forward(self, x):
        h = self.pool(F.relu(self.conv1(x)))
        h = self.pool(F.relu(self.conv2(h)))
        h = h.view(-1, 16 * 5 * 5)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        h1 = h[:, :-self.num_rej]
        h2 = h[:, -self.num_rej:]
        return h, h1, h2

    def pconf(self, hidden, t, loss, rate_all=1.0, rate_inv=0.8):
        for i in range(self.num_rej):
            if i == 0:
                h = hidden[:, i]
                loss += torch.mean(self.loss(h) + (1 - rate_all) / rate_all *
                    self.loss(-h))
            else:
                h = hidden[:, i]
                temp = (t == i).type(torch.FloatTensor)
                if type(rate_inv) == float:
                    temp *= rate_inv
                else:
                    temp *= rate_inv[i]
                loss += torch.mean((self.loss(h) + (1 - rate_inv) /
                    rate_inv * self.loss(-h)) * temp)
        return loss


def get_inputs():
    return [torch.rand([4, 3, 32, 32])]


def get_init_inputs():
    return [[], {}]
