import torch
import torch.nn as nn
import torch.autograd


class conv_head_pooling(nn.Module):

    def __init__(self, in_feature, out_feature, stride, padding_mode='zeros'):
        super(conv_head_pooling, self).__init__()
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.avgpool = nn.AvgPool2d(3, 2, 1)
        self.conv1 = nn.Conv2d(in_feature, out_feature, 1, 1)
        self.conv2 = nn.Conv2d(in_feature, out_feature, 1, 1)
        self.conv3 = nn.Conv2d(2 * out_feature, out_feature, 1, 1)
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x, cls_token):
        max = self.maxpool(x)
        min = -self.maxpool(-x)
        avg = self.avgpool(x)
        avg = avg - max - min
        max_2 = self.conv1(max)
        avg_2 = self.conv2(max)
        x = torch.cat([avg_2, max_2], dim=1)
        x = self.conv3(x)
        x = x + max_2
        cls_token = self.fc(cls_token)
        return x, cls_token


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_feature': 4, 'out_feature': 4, 'stride': 1}]
