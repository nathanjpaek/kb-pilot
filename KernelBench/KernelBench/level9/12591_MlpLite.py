import torch
from torch import nn


class MlpLite(nn.Module):

    def __init__(self, H, W, in_features, hidden_features=None,
        out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        in_c = H * W
        self.H = H
        self.W = W
        self.in_c = in_c
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1x1 = nn.Conv2d(out_features, out_features, kernel_size=1,
            stride=1)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.view(-1, self.H, self.W, self.in_features).permute(0, 3, 1, 2)
        x = self.avgpool(x)
        x = x.permute(0, 3, 1, 2).reshape(-1, self.in_c // 4, self.in_features)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = x.view(-1, self.H // 2, self.W // 2, self.in_features).permute(
            0, 3, 1, 2)
        x = self.upsample(x)
        x = self.conv1x1(x)
        x = x.permute(0, 3, 1, 2).reshape(-1, self.in_c, self.in_features)
        x = self.drop(x)
        return x

    def flops(self):
        flops = 0
        flops += 2 * self.in_features * self.hidden_features * self.in_c // 4
        flops += self.in_c * self.in_features
        flops += self.in_c * self.in_features * self.in_features
        return flops


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'H': 4, 'W': 4, 'in_features': 4}]
