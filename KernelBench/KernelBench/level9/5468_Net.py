import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.p_fc1 = nn.Conv2d(128, 4, 1)
        self.p_fc2 = nn.Linear(4 * 8 * 8, 64)
        self.v_fc1 = nn.Conv2d(128, 2, 1)
        self.v_fc2 = nn.Linear(2 * 8 * 8, 1)

    def forward(self, x):
        a1 = F.relu(self.conv1(x))
        a2 = F.relu(self.conv2(a1))
        a3 = F.relu(self.conv3(a2))
        p1 = F.relu(self.p_fc1(a3))
        p_act = p1.view(-1, 4 * 8 * 8)
        p_out = F.softmax(self.p_fc2(p_act), dim=0)
        v1 = F.relu(self.v_fc1(a3))
        v_act = v1.view(-1, 2 * 8 * 8)
        v_out = torch.tanh(self.v_fc2(v_act))
        return p_out, v_out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
