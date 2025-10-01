import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, input_size, action_dim, conv=False, conv_size=16,
        fc_size=32, K=2):
        super(Model, self).__init__()
        self.input_size = input_size
        self.input_h = int(np.sqrt(input_size))
        self.action_dim = action_dim
        self.K = K
        self.conv = conv
        self.conv_size = conv_size
        self.fc_size = fc_size
        if self.conv:
            self.conv1 = nn.Conv2d(1, self.conv_size, kernel_size=3, stride=1)
            self.conv2 = nn.Conv2d(self.conv_size, self.conv_size,
                kernel_size=3, stride=1)
            self.conv3 = nn.Conv2d(self.conv_size, self.conv_size,
                kernel_size=3, stride=1)
            self.conv4 = nn.Conv2d(self.conv_size, self.conv_size,
                kernel_size=3, stride=1)
            self.fc = nn.Linear(2 * 2 * self.conv_size + self.action_dim,
                self.fc_size)
        else:
            self.fc = nn.Linear(self.input_size + self.action_dim, self.fc_size
                )
        self.rew_out = nn.Linear(self.fc_size, 1)
        self.pred_out = nn.Linear(self.fc_size, self.input_size)

    def forward(self, x, a):
        if self.conv:
            out = x.unsqueeze(1)
            out = F.relu(self.conv1(out))
            out = F.relu(self.conv2(out))
            out = F.relu(self.conv3(out))
            out = F.relu(self.conv4(out))
            out = out.view(out.size(0), -1)
            out = torch.cat((out, a), dim=-1)
        else:
            out = torch.cat((x, a), dim=-1)
        out = F.relu(self.fc(out))
        return self.pred_out(out).reshape(out.size(0), self.input_h, self.
            input_h), torch.sigmoid(self.rew_out(out))


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'action_dim': 4}]
