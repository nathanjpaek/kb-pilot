import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv5 = nn.Conv2d(256, 512, 3, 2, 1)
        self.fc1 = nn.Linear(1 * 1 * 512, 4 * 4 * 256)
        self.conv6 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv7 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv8 = nn.Conv2d(512, 512, 3, 1, 1)
        self.branch1_fc1 = nn.Linear(512 * 32 * 32, 32)
        self.branch1_fc2 = nn.Linear(32, 32)
        self.branch1_fc3 = nn.Linear(32, 32 * 32 * 300)
        self.branch2_fc1 = nn.Linear(512 * 32 * 32, 32)
        self.branch2_fc2 = nn.Linear(32, 32)
        self.branch2_fc3 = nn.Linear(32, 32 * 32 * 100)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 8)
        x = x.view(-1, 1 * 1 * 512)
        x = F.relu(self.fc1(x))
        x = x.view(-1, 256, 4, 4)
        x = F.relu(self.conv6(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners
            =False)
        x = F.relu(self.conv7(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners
            =False)
        x = F.relu(self.conv8(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners
            =False)
        x = x.view(-1, 512 * 32 * 32)
        branch1_x = F.relu(self.branch1_fc1(x))
        branch1_x = F.relu(self.branch1_fc2(branch1_x))
        branch1_x = F.relu(self.branch1_fc3(branch1_x))
        branch1_x = branch1_x.view(-1, 100, 3, 32, 32)
        branch2_x = F.relu(self.branch2_fc1(x))
        branch2_x = F.relu(self.branch2_fc2(branch2_x))
        branch2_x = F.relu(self.branch2_fc3(branch2_x))
        branch2_x = branch2_x.view(-1, 100, 1, 32, 32)
        x = [branch1_x, branch2_x]
        return torch.cat(x, 2)


def get_inputs():
    return [torch.rand([4, 3, 256, 256])]


def get_init_inputs():
    return [[], {}]
