import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        self.branch_pool = BasicConv2d(in_channels, pool_features,
            kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)
        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=
            (0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding
            =(3, 0))
        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1),
            padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7),
            padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1),
            padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7),
            padding=(0, 3))
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)
        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7),
            padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1),
            padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)
        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3),
            padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1),
            padding=(1, 0))
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3),
            padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1),
            padding=(1, 0))
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)
            ]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.
            branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class Inception3(nn.Module):

    def __init__(self, num_classes=1000, transform_input=True):
        super(Inception3, self).__init__()
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.fc = nn.Linear(2048, num_classes)

    def set_params(self, dict):
        self.Conv2d_1a_3x3.conv.weight = nn.Parameter(torch.FloatTensor(
            dict['layer1']['c_1_w']))
        self.Conv2d_1a_3x3.conv.bias = nn.Parameter(torch.FloatTensor(dict[
            'layer1']['c_1_b']))
        self.Conv2d_2a_3x3.conv.weight = nn.Parameter(torch.FloatTensor(
            dict['layer1']['c_2_w']))
        self.Conv2d_2a_3x3.conv.bias = nn.Parameter(torch.FloatTensor(dict[
            'layer1']['c_2_b']))
        self.Conv2d_2b_3x3.conv.weight = nn.Parameter(torch.FloatTensor(
            dict['layer1']['c_3_w']))
        self.Conv2d_2b_3x3.conv.bias = nn.Parameter(torch.FloatTensor(dict[
            'layer1']['c_3_b']))
        self.Conv2d_3b_1x1.conv.weight = nn.Parameter(torch.FloatTensor(
            dict['layer1']['c_4_w']))
        self.Conv2d_3b_1x1.conv.bias = nn.Parameter(torch.FloatTensor(dict[
            'layer1']['c_4_b']))
        self.Conv2d_4a_3x3.conv.weight = nn.Parameter(torch.FloatTensor(
            dict['layer1']['c_5_w']))
        self.Conv2d_4a_3x3.conv.bias = nn.Parameter(torch.FloatTensor(dict[
            'layer1']['c_5_b']))
        self.Mixed_5b.branch1x1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer2_1']['way1_w']))
        self.Mixed_5b.branch1x1.conv.bias = nn.Parameter(torch.FloatTensor(
            dict['layer2_1']['way1_b']))
        self.Mixed_5b.branch5x5_1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer2_1']['way2_1_w']))
        self.Mixed_5b.branch5x5_1.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer2_1']['way2_1_b']))
        self.Mixed_5b.branch5x5_2.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer2_1']['way2_2_w']))
        self.Mixed_5b.branch5x5_2.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer2_1']['way2_2_b']))
        self.Mixed_5b.branch3x3dbl_1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer2_1']['way3_1_w']))
        self.Mixed_5b.branch3x3dbl_1.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer2_1']['way3_1_b']))
        self.Mixed_5b.branch3x3dbl_2.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer2_1']['way3_2_w']))
        self.Mixed_5b.branch3x3dbl_2.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer2_1']['way3_2_b']))
        self.Mixed_5b.branch3x3dbl_3.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer2_1']['way3_3_w']))
        self.Mixed_5b.branch3x3dbl_3.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer2_1']['way3_3_b']))
        self.Mixed_5b.branch_pool.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer2_1']['way4_w']))
        self.Mixed_5b.branch_pool.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer2_1']['way4_b']))
        self.Mixed_5c.branch1x1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer2_2']['way1_w']))
        self.Mixed_5c.branch1x1.conv.bias = nn.Parameter(torch.FloatTensor(
            dict['layer2_2']['way1_b']))
        self.Mixed_5c.branch5x5_1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer2_2']['way2_1_w']))
        self.Mixed_5c.branch5x5_1.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer2_2']['way2_1_b']))
        self.Mixed_5c.branch5x5_2.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer2_2']['way2_2_w']))
        self.Mixed_5c.branch5x5_2.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer2_2']['way2_2_b']))
        self.Mixed_5c.branch3x3dbl_1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer2_2']['way3_1_w']))
        self.Mixed_5c.branch3x3dbl_1.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer2_2']['way3_1_b']))
        self.Mixed_5c.branch3x3dbl_2.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer2_2']['way3_2_w']))
        self.Mixed_5c.branch3x3dbl_2.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer2_2']['way3_2_b']))
        self.Mixed_5c.branch3x3dbl_3.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer2_2']['way3_3_w']))
        self.Mixed_5c.branch3x3dbl_3.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer2_2']['way3_3_b']))
        self.Mixed_5c.branch_pool.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer2_2']['way4_w']))
        self.Mixed_5c.branch_pool.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer2_2']['way4_b']))
        self.Mixed_5d.branch1x1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer2_3']['way1_w']))
        self.Mixed_5d.branch1x1.conv.bias = nn.Parameter(torch.FloatTensor(
            dict['layer2_3']['way1_b']))
        self.Mixed_5d.branch5x5_1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer2_3']['way2_1_w']))
        self.Mixed_5d.branch5x5_1.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer2_3']['way2_1_b']))
        self.Mixed_5d.branch5x5_2.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer2_3']['way2_2_w']))
        self.Mixed_5d.branch5x5_2.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer2_3']['way2_2_b']))
        self.Mixed_5d.branch3x3dbl_1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer2_3']['way3_1_w']))
        self.Mixed_5d.branch3x3dbl_1.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer2_3']['way3_1_b']))
        self.Mixed_5d.branch3x3dbl_2.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer2_3']['way3_2_w']))
        self.Mixed_5d.branch3x3dbl_2.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer2_3']['way3_2_b']))
        self.Mixed_5d.branch3x3dbl_3.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer2_3']['way3_3_w']))
        self.Mixed_5d.branch3x3dbl_3.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer2_3']['way3_3_b']))
        self.Mixed_5d.branch_pool.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer2_3']['way4_w']))
        self.Mixed_5d.branch_pool.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer2_3']['way4_b']))
        self.Mixed_6a.branch3x3.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer3']['way1_w']))
        self.Mixed_6a.branch3x3.conv.bias = nn.Parameter(torch.FloatTensor(
            dict['layer3']['way1_b']))
        self.Mixed_6a.branch3x3dbl_1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer3']['way2_1_w']))
        self.Mixed_6a.branch3x3dbl_1.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer3']['way2_1_b']))
        self.Mixed_6a.branch3x3dbl_2.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer3']['way2_2_w']))
        self.Mixed_6a.branch3x3dbl_2.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer3']['way2_2_b']))
        self.Mixed_6a.branch3x3dbl_3.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer3']['way2_3_w']))
        self.Mixed_6a.branch3x3dbl_3.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer3']['way2_3_b']))
        self.Mixed_6b.branch1x1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_1']['way1_w']))
        self.Mixed_6b.branch1x1.conv.bias = nn.Parameter(torch.FloatTensor(
            dict['layer4_1']['way1_b']))
        self.Mixed_6b.branch7x7_1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_1']['way2_1_w']))
        self.Mixed_6b.branch7x7_1.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_1']['way2_1_b']))
        self.Mixed_6b.branch7x7_2.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_1']['way2_2_w']))
        self.Mixed_6b.branch7x7_2.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_1']['way2_2_b']))
        self.Mixed_6b.branch7x7_3.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_1']['way2_3_w']))
        self.Mixed_6b.branch7x7_3.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_1']['way2_3_b']))
        self.Mixed_6b.branch7x7dbl_1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_1']['way3_1_w']))
        self.Mixed_6b.branch7x7dbl_1.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_1']['way3_1_b']))
        self.Mixed_6b.branch7x7dbl_2.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_1']['way3_2_w']))
        self.Mixed_6b.branch7x7dbl_2.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_1']['way3_2_b']))
        self.Mixed_6b.branch7x7dbl_3.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_1']['way3_3_w']))
        self.Mixed_6b.branch7x7dbl_3.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_1']['way3_3_b']))
        self.Mixed_6b.branch7x7dbl_4.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_1']['way3_4_w']))
        self.Mixed_6b.branch7x7dbl_4.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_1']['way3_4_b']))
        self.Mixed_6b.branch7x7dbl_5.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_1']['way3_5_w']))
        self.Mixed_6b.branch7x7dbl_5.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_1']['way3_5_b']))
        self.Mixed_6b.branch_pool.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_1']['way4_w']))
        self.Mixed_6b.branch_pool.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_1']['way4_b']))
        self.Mixed_6c.branch1x1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_2']['way1_w']))
        self.Mixed_6c.branch1x1.conv.bias = nn.Parameter(torch.FloatTensor(
            dict['layer4_2']['way1_b']))
        self.Mixed_6c.branch7x7_1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_2']['way2_1_w']))
        self.Mixed_6c.branch7x7_1.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_2']['way2_1_b']))
        self.Mixed_6c.branch7x7_2.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_2']['way2_2_w']))
        self.Mixed_6c.branch7x7_2.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_2']['way2_2_b']))
        self.Mixed_6c.branch7x7_3.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_2']['way2_3_w']))
        self.Mixed_6c.branch7x7_3.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_2']['way2_3_b']))
        self.Mixed_6c.branch7x7dbl_1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_2']['way3_1_w']))
        self.Mixed_6c.branch7x7dbl_1.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_2']['way3_1_b']))
        self.Mixed_6c.branch7x7dbl_2.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_2']['way3_2_w']))
        self.Mixed_6c.branch7x7dbl_2.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_2']['way3_2_b']))
        self.Mixed_6c.branch7x7dbl_3.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_2']['way3_3_w']))
        self.Mixed_6c.branch7x7dbl_3.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_2']['way3_3_b']))
        self.Mixed_6c.branch7x7dbl_4.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_2']['way3_4_w']))
        self.Mixed_6c.branch7x7dbl_4.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_2']['way3_4_b']))
        self.Mixed_6c.branch7x7dbl_5.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_2']['way3_5_w']))
        self.Mixed_6c.branch7x7dbl_5.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_2']['way3_5_b']))
        self.Mixed_6c.branch_pool.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_2']['way4_w']))
        self.Mixed_6c.branch_pool.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_2']['way4_b']))
        self.Mixed_6d.branch1x1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_3']['way1_w']))
        self.Mixed_6d.branch1x1.conv.bias = nn.Parameter(torch.FloatTensor(
            dict['layer4_3']['way1_b']))
        self.Mixed_6d.branch7x7_1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_3']['way2_1_w']))
        self.Mixed_6d.branch7x7_1.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_3']['way2_1_b']))
        self.Mixed_6d.branch7x7_2.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_3']['way2_2_w']))
        self.Mixed_6d.branch7x7_2.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_3']['way2_2_b']))
        self.Mixed_6d.branch7x7_3.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_3']['way2_3_w']))
        self.Mixed_6d.branch7x7_3.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_3']['way2_3_b']))
        self.Mixed_6d.branch7x7dbl_1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_3']['way3_1_w']))
        self.Mixed_6d.branch7x7dbl_1.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_3']['way3_1_b']))
        self.Mixed_6d.branch7x7dbl_2.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_3']['way3_2_w']))
        self.Mixed_6d.branch7x7dbl_2.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_3']['way3_2_b']))
        self.Mixed_6d.branch7x7dbl_3.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_3']['way3_3_w']))
        self.Mixed_6d.branch7x7dbl_3.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_3']['way3_3_b']))
        self.Mixed_6d.branch7x7dbl_4.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_3']['way3_4_w']))
        self.Mixed_6d.branch7x7dbl_4.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_3']['way3_4_b']))
        self.Mixed_6d.branch7x7dbl_5.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_3']['way3_5_w']))
        self.Mixed_6d.branch7x7dbl_5.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_3']['way3_5_b']))
        self.Mixed_6d.branch_pool.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_3']['way4_w']))
        self.Mixed_6d.branch_pool.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_3']['way4_b']))
        self.Mixed_6e.branch1x1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_4']['way1_w']))
        self.Mixed_6e.branch1x1.conv.bias = nn.Parameter(torch.FloatTensor(
            dict['layer4_4']['way1_b']))
        self.Mixed_6e.branch7x7_1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_4']['way2_1_w']))
        self.Mixed_6e.branch7x7_1.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_4']['way2_1_b']))
        self.Mixed_6e.branch7x7_2.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_4']['way2_2_w']))
        self.Mixed_6e.branch7x7_2.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_4']['way2_2_b']))
        self.Mixed_6e.branch7x7_3.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_4']['way2_3_w']))
        self.Mixed_6e.branch7x7_3.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_4']['way2_3_b']))
        self.Mixed_6e.branch7x7dbl_1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_4']['way3_1_w']))
        self.Mixed_6e.branch7x7dbl_1.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_4']['way3_1_b']))
        self.Mixed_6e.branch7x7dbl_2.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_4']['way3_2_w']))
        self.Mixed_6e.branch7x7dbl_2.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_4']['way3_2_b']))
        self.Mixed_6e.branch7x7dbl_3.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_4']['way3_3_w']))
        self.Mixed_6e.branch7x7dbl_3.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_4']['way3_3_b']))
        self.Mixed_6e.branch7x7dbl_4.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_4']['way3_4_w']))
        self.Mixed_6e.branch7x7dbl_4.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_4']['way3_4_b']))
        self.Mixed_6e.branch7x7dbl_5.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_4']['way3_5_w']))
        self.Mixed_6e.branch7x7dbl_5.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_4']['way3_5_b']))
        self.Mixed_6e.branch_pool.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer4_4']['way4_w']))
        self.Mixed_6e.branch_pool.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer4_4']['way4_b']))
        self.Mixed_7a.branch3x3_1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer5']['way1_1_w']))
        self.Mixed_7a.branch3x3_1.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer5']['way1_1_b']))
        self.Mixed_7a.branch3x3_2.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer5']['way1_2_w']))
        self.Mixed_7a.branch3x3_2.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer5']['way1_2_b']))
        self.Mixed_7a.branch7x7x3_1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer5']['way2_1_w']))
        self.Mixed_7a.branch7x7x3_1.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer5']['way2_1_b']))
        self.Mixed_7a.branch7x7x3_2.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer5']['way2_2_w']))
        self.Mixed_7a.branch7x7x3_2.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer5']['way2_2_b']))
        self.Mixed_7a.branch7x7x3_3.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer5']['way2_3_w']))
        self.Mixed_7a.branch7x7x3_3.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer5']['way2_3_b']))
        self.Mixed_7a.branch7x7x3_4.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer5']['way2_4_w']))
        self.Mixed_7a.branch7x7x3_4.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer5']['way2_4_b']))
        self.Mixed_7b.branch1x1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer6_1']['way1_w']))
        self.Mixed_7b.branch1x1.conv.bias = nn.Parameter(torch.FloatTensor(
            dict['layer6_1']['way1_b']))
        self.Mixed_7b.branch3x3_1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer6_1']['way23_1_w']))
        self.Mixed_7b.branch3x3_1.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer6_1']['way23_1_b']))
        self.Mixed_7b.branch3x3_2a.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer6_1']['way2_2_w']))
        self.Mixed_7b.branch3x3_2a.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer6_1']['way2_2_b']))
        self.Mixed_7b.branch3x3_2b.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer6_1']['way3_2_w']))
        self.Mixed_7b.branch3x3_2b.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer6_1']['way3_2_b']))
        self.Mixed_7b.branch3x3dbl_1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer6_1']['way45_1_w']))
        self.Mixed_7b.branch3x3dbl_1.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer6_1']['way45_1_b']))
        self.Mixed_7b.branch3x3dbl_2.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer6_1']['way45_2_w']))
        self.Mixed_7b.branch3x3dbl_2.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer6_1']['way45_2_b']))
        self.Mixed_7b.branch3x3dbl_3a.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer6_1']['way4_3_w']))
        self.Mixed_7b.branch3x3dbl_3a.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer6_1']['way4_3_b']))
        self.Mixed_7b.branch3x3dbl_3b.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer6_1']['way5_3_w']))
        self.Mixed_7b.branch3x3dbl_3b.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer6_1']['way5_3_b']))
        self.Mixed_7b.branch_pool.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer6_1']['way6_w']))
        self.Mixed_7b.branch_pool.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer6_1']['way6_b']))
        self.Mixed_7c.branch1x1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer6_2']['way1_w']))
        self.Mixed_7c.branch1x1.conv.bias = nn.Parameter(torch.FloatTensor(
            dict['layer6_2']['way1_b']))
        self.Mixed_7c.branch3x3_1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer6_2']['way23_1_w']))
        self.Mixed_7c.branch3x3_1.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer6_2']['way23_1_b']))
        self.Mixed_7c.branch3x3_2a.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer6_2']['way2_2_w']))
        self.Mixed_7c.branch3x3_2a.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer6_2']['way2_2_b']))
        self.Mixed_7c.branch3x3_2b.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer6_2']['way3_2_w']))
        self.Mixed_7c.branch3x3_2b.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer6_2']['way3_2_b']))
        self.Mixed_7c.branch3x3dbl_1.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer6_2']['way45_1_w']))
        self.Mixed_7c.branch3x3dbl_1.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer6_2']['way45_1_b']))
        self.Mixed_7c.branch3x3dbl_2.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer6_2']['way45_2_w']))
        self.Mixed_7c.branch3x3dbl_2.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer6_2']['way45_2_b']))
        self.Mixed_7c.branch3x3dbl_3a.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer6_2']['way4_3_w']))
        self.Mixed_7c.branch3x3dbl_3a.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer6_2']['way4_3_b']))
        self.Mixed_7c.branch3x3dbl_3b.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer6_2']['way5_3_w']))
        self.Mixed_7c.branch3x3dbl_3b.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer6_2']['way5_3_b']))
        self.Mixed_7c.branch_pool.conv.weight = nn.Parameter(torch.
            FloatTensor(dict['layer6_2']['way6_w']))
        self.Mixed_7c.branch_pool.conv.bias = nn.Parameter(torch.
            FloatTensor(dict['layer6_2']['way6_b']))
        self.fc.weight = nn.Parameter(torch.FloatTensor(dict['outputlayer']
            ['fc_w']))
        self.fc.bias = nn.Parameter(torch.FloatTensor(dict['outputlayer'][
            'fc_b']))

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0, :, :] = x[:, 0, :, :] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1, :, :] = x[:, 1, :, :] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2, :, :] = x[:, 2, :, :] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = F.avg_pool2d(x, kernel_size=8)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 512, 512])]


def get_init_inputs():
    return [[], {}]
