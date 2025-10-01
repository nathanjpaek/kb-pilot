import torch
import torch.nn as nn


def crop(image, new_shape):
    plus_h, plus_w = 0, 0
    if new_shape[2] % 2 != 0:
        plus_h = 1
    if new_shape[3] % 2 != 0:
        plus_w = 1
    middle_height = image.shape[2] // 2
    middle_weight = image.shape[3] // 2
    go_height = new_shape[2] // 2
    go_weight = new_shape[3] // 2
    cropped_image = image[:, :, middle_height - go_height:middle_height +
        go_height + plus_h, middle_weight - go_weight:middle_weight +
        go_weight + plus_w]
    return cropped_image


class ContractingBlock(nn.Module):

    def __init__(self, input_channel):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=
            input_channel * 2, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(input_channel * 2, input_channel * 2,
            kernel_size=(3, 3))
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        return self.maxpool(x)


class ExpandingBlock(nn.Module):

    def __init__(self, input_channels):
        super(ExpandingBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
            align_corners=True)
        self.conv1 = nn.Conv2d(input_channels, input_channels // 2,
            kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(input_channels, input_channels // 2,
            kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(input_channels // 2, input_channels // 2,
            kernel_size=(3, 3))
        self.activation = nn.ReLU()

    def forward(self, skip_con_x, x):
        x = self.upsample(x)
        x = self.conv1(x)
        skip_con_x = crop(skip_con_x, x.shape)
        x = torch.cat([skip_con_x, x], axis=1)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        return self.activation(x)


class FeatureMapBlock(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=
            (1, 1))

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):

    def __init__(self, input_channels, output_channels, hidden_channels=64):
        super(Unet, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        self.contract3 = ContractingBlock(hidden_channels * 4)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        self.expanding1 = ExpandingBlock(hidden_channels * 16)
        self.expanding2 = ExpandingBlock(hidden_channels * 8)
        self.expanding3 = ExpandingBlock(hidden_channels * 4)
        self.expanding4 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)

    def forward(self, x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        x5 = self.expanding1(skip_con_x=x3, x=x4)
        x6 = self.expanding2(skip_con_x=x2, x=x5)
        x7 = self.expanding3(skip_con_x=x1, x=x6)
        x8 = self.expanding4(skip_con_x=x0, x=x7)
        return self.downfeature(x8)


def get_inputs():
    return [torch.rand([4, 4, 256, 256])]


def get_init_inputs():
    return [[], {'input_channels': 4, 'output_channels': 4}]
