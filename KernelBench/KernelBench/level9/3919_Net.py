import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv2 = nn.Conv2d(3, 64, 8, 2, 3)
        self.conv3 = nn.Conv2d(64, 128, 6, 2, 2)
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv5 = nn.Conv2d(256, 512, 4, 2, 1)
        self.fc1 = nn.Linear(512, 4096)
        self.conv6 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv7 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv8 = nn.Conv2d(512, 512, 3, 1, 1)
        self.branch1_fc1 = nn.Conv2d(512, 512, 1, 1, 0)
        self.branch1_fc2 = nn.Conv2d(512, 512, 1, 1, 0)
        self.branch1_fc3 = nn.Conv2d(512, 300, 1, 1, 0)
        self.branch2_fc1 = nn.Conv2d(512, 512, 1, 1, 0)
        self.branch2_fc2 = nn.Conv2d(512, 512, 1, 1, 0)
        self.branch2_fc3 = nn.Conv2d(512, 100, 1, 1, 0)

    def forward(self, x, interp_factor=1):
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.tanh(F.max_pool2d(x, 8))
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
        if interp_factor != 1:
            x = F.interpolate(x, scale_factor=interp_factor, mode=
                'bilinear', align_corners=False)
        branch1_x = F.relu(self.branch1_fc1(x))
        branch1_x = F.relu(self.branch1_fc2(branch1_x))
        branch1_x = self.branch1_fc3(branch1_x)
        branch1_x = branch1_x.view(-1, 100, 3, 32 * interp_factor, 32 *
            interp_factor)
        branch2_x = F.relu(self.branch2_fc1(x))
        branch2_x = F.relu(self.branch2_fc2(branch2_x))
        branch2_x = self.branch2_fc3(branch2_x)
        branch2_x = branch2_x.view(-1, 100, 1, 32 * interp_factor, 32 *
            interp_factor)
        x = torch.cat([branch1_x, branch2_x], 2)
        return x


def get_inputs():
    return [torch.rand([4, 3, 128, 128])]


def get_init_inputs():
    return [[], {}]
