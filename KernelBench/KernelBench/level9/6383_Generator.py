import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear',
            align_corners=False)
        self.conv7_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv7_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear',
            align_corners=False)
        self.conv8_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv8_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear',
            align_corners=False)
        self.conv9_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear',
            align_corners=False)
        self.conv10_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv10_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv10_3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, xb):
        xb = F.relu(self.conv6_1(xb))
        xb = F.relu(self.conv6_2(xb))
        xb = F.relu(self.conv6_3(xb))
        xb = self.upsample1(xb)
        xb = F.relu(self.conv7_1(xb))
        xb = F.relu(self.conv7_2(xb))
        xb = F.relu(self.conv7_3(xb))
        xb = self.upsample2(xb)
        xb = F.relu(self.conv8_1(xb))
        xb = F.relu(self.conv8_2(xb))
        xb = F.relu(self.conv8_3(xb))
        xb = self.upsample3(xb)
        xb = F.relu(self.conv9_1(xb))
        xb = F.relu(self.conv9_2(xb))
        xb = self.upsample4(xb)
        xb = F.relu(self.conv10_1(xb))
        xb = F.relu(self.conv10_2(xb))
        xb = self.conv10_3(xb)
        return xb


class Encoder(nn.Module):

    def __init__(self, model_dict=None):
        super(Encoder, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        if model_dict is not None:
            self.conv1_1.weight.data = model_dict['conv1_1_weight']
            self.conv1_1.bias.data = model_dict['conv1_1_bias']
            self.conv1_2.weight.data = model_dict['conv1_2_weight']
            self.conv1_2.bias.data = model_dict['conv1_2_bias']
            self.conv2_1.weight.data = model_dict['conv2_1_weight']
            self.conv2_1.bias.data = model_dict['conv2_1_bias']
            self.conv2_2.weight.data = model_dict['conv2_2_weight']
            self.conv2_2.bias.data = model_dict['conv2_2_bias']
            self.conv3_1.weight.data = model_dict['conv3_1_weight']
            self.conv3_1.bias.data = model_dict['conv3_1_bias']
            self.conv3_2.weight.data = model_dict['conv3_2_weight']
            self.conv3_2.bias.data = model_dict['conv3_2_bias']
            self.conv3_3.weight.data = model_dict['conv3_3_weight']
            self.conv3_3.bias.data = model_dict['conv3_3_bias']
            self.conv4_1.weight.data = model_dict['conv4_1_weight']
            self.conv4_1.bias.data = model_dict['conv4_1_bias']
            self.conv4_2.weight.data = model_dict['conv4_2_weight']
            self.conv4_2.bias.data = model_dict['conv4_2_bias']
            self.conv4_3.weight.data = model_dict['conv4_3_weight']
            self.conv4_3.bias.data = model_dict['conv4_3_bias']
            self.conv5_1.weight.data = model_dict['conv5_1_weight']
            self.conv5_1.bias.data = model_dict['conv5_1_bias']
            self.conv5_2.weight.data = model_dict['conv5_2_weight']
            self.conv5_2.bias.data = model_dict['conv5_2_bias']
            self.conv5_3.weight.data = model_dict['conv5_3_weight']
            self.conv5_3.bias.data = model_dict['conv5_3_bias']

    def forward(self, xb):
        xb = F.relu(self.conv1_1(xb))
        xb = F.relu(self.conv1_2(xb))
        xb = self.pool1(xb)
        xb = F.relu(self.conv2_1(xb))
        xb = F.relu(self.conv2_2(xb))
        xb = self.pool2(xb)
        xb = F.relu(self.conv3_1(xb))
        xb = F.relu(self.conv3_2(xb))
        xb = F.relu(self.conv3_3(xb))
        xb = self.pool3(xb)
        xb = F.relu(self.conv4_1(xb))
        xb = F.relu(self.conv4_2(xb))
        xb = F.relu(self.conv4_3(xb))
        xb = self.pool4(xb)
        xb = F.relu(self.conv5_1(xb))
        xb = F.relu(self.conv5_2(xb))
        xb = F.relu(self.conv5_3(xb))
        return xb


class Generator(nn.Module):

    def __init__(self, model_dict=None):
        super(Generator, self).__init__()
        self.encoder = Encoder(model_dict)
        self.decoder = Decoder()

    def forward(self, xb):
        xb = self.encoder(xb)
        xb = self.decoder(xb)
        return xb


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
