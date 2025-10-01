import torch
import torch.nn as nn
import torch.nn.functional as F


class resnet_block(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(resnet_block, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        if self.dim_in == self.dim_out:
            self.conv_1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=1,
                padding=1, bias=False)
            self.bn_1 = nn.InstanceNorm2d(self.dim_out)
            self.conv_2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1,
                padding=1, bias=False)
            self.bn_2 = nn.InstanceNorm2d(self.dim_out)
        else:
            self.conv_1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=2,
                padding=1, bias=False)
            self.bn_1 = nn.InstanceNorm2d(self.dim_out)
            self.conv_2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1,
                padding=1, bias=False)
            self.bn_2 = nn.InstanceNorm2d(self.dim_out)
            self.conv_s = nn.Conv2d(self.dim_in, self.dim_out, 1, stride=2,
                padding=0, bias=False)
            self.bn_s = nn.InstanceNorm2d(self.dim_out)

    def forward(self, input):
        if self.dim_in == self.dim_out:
            output = self.bn_1(self.conv_1(input))
            output = F.leaky_relu(output, negative_slope=0.02, inplace=True)
            output = self.bn_2(self.conv_2(output))
            output = output + input
            output = F.leaky_relu(output, negative_slope=0.02, inplace=True)
        else:
            output = self.bn_1(self.conv_1(input))
            output = F.leaky_relu(output, negative_slope=0.02, inplace=True)
            output = self.bn_2(self.conv_2(output))
            input_ = self.bn_s(self.conv_s(input))
            output = output + input_
            output = F.leaky_relu(output, negative_slope=0.02, inplace=True)
        return output


class img_encoder(nn.Module):

    def __init__(self, img_ef_dim, z_dim):
        super(img_encoder, self).__init__()
        self.img_ef_dim = img_ef_dim
        self.z_dim = z_dim
        self.conv_0 = nn.Conv2d(1, self.img_ef_dim, 7, stride=2, padding=3,
            bias=False)
        self.bn_0 = nn.InstanceNorm2d(self.img_ef_dim)
        self.res_1 = resnet_block(self.img_ef_dim, self.img_ef_dim)
        self.res_2 = resnet_block(self.img_ef_dim, self.img_ef_dim)
        self.res_3 = resnet_block(self.img_ef_dim, self.img_ef_dim * 2)
        self.res_4 = resnet_block(self.img_ef_dim * 2, self.img_ef_dim * 2)
        self.res_5 = resnet_block(self.img_ef_dim * 2, self.img_ef_dim * 4)
        self.res_6 = resnet_block(self.img_ef_dim * 4, self.img_ef_dim * 4)
        self.conv_9 = nn.Conv2d(self.img_ef_dim * 4, self.img_ef_dim * 4, 4,
            stride=2, padding=1, bias=False)
        self.bn_9 = nn.InstanceNorm2d(self.img_ef_dim * 4)
        self.conv_10 = nn.Conv2d(self.img_ef_dim * 4, self.z_dim, 4, stride
            =1, padding=0, bias=True)

    def forward(self, view):
        layer_0 = self.bn_0(self.conv_0(1 - view))
        layer_0 = F.leaky_relu(layer_0, negative_slope=0.02, inplace=True)
        layer_1 = self.res_1(layer_0)
        layer_2 = self.res_2(layer_1)
        layer_3 = self.res_3(layer_2)
        layer_4 = self.res_4(layer_3)
        layer_5 = self.res_5(layer_4)
        layer_6 = self.res_6(layer_5)
        layer_9 = self.bn_9(self.conv_9(layer_6))
        layer_9 = F.leaky_relu(layer_9, negative_slope=0.02, inplace=True)
        layer_10 = self.conv_10(layer_9)
        layer_10 = layer_10.view(-1)
        return layer_10


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {'img_ef_dim': 4, 'z_dim': 4}]
