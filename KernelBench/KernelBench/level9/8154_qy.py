import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity
            ='leaky_relu')
        try:
            nn.init.constant_(m.bias, 0.01)
        except:
            pass
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity
            ='leaky_relu')
        nn.init.constant_(m.bias, 0.01)


class Generator(nn.Module):

    def __init__(self, input_dim=8, output_dim=2):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, output_dim)
        self.apply(weights_init)

    def forward(self, condition, v0, t):
        x = torch.cat([condition, v0], dim=1)
        x = torch.cat([x, t], dim=1)
        x = self.linear1(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.linear2(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.linear3(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.linear4(x)
        x = torch.cos(x)
        x = self.linear5(x)
        return x


class qy(nn.Module):

    def __init__(self, zy_dim):
        super(qy, self).__init__()
        self.trajectory_generation = Generator(input_dim=1 + 1 + zy_dim,
            output_dim=4)

    def forward(self, zy, v0, t):
        h = F.leaky_relu(zy, inplace=True)
        condition = h.unsqueeze(1)
        condition = condition.expand(h.shape[0], t.shape[-1], h.shape[-1])
        condition = condition.reshape(h.shape[0] * t.shape[-1], h.shape[-1])
        output = self.trajectory_generation(condition, v0.view(-1, 1), t.
            view(-1, 1))
        output_xy = output[:, :2]
        logvar = output[:, 2:]
        return output_xy, logvar


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'zy_dim': 4}]
