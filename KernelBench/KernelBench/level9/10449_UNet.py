import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class UNet(nn.Module):
    """
    The U-Net Convolutional Neural Network for semantic segmentation

    Source material for the algorithm:
    https://link.springer.com/chapter/10.1007%2F978-3-319-24574-4_28
    """

    def __init__(self, in_channels=1, num_classes=2, pretrained=False,
        padding=1):
        """
        Arguments:
            in_channels (int): the number of channels in the input image
            num_classes (int, optional): the number of classes to be predicted.
                Note: The background is always considered as 1 class.
                (By default predicts the background and foreground: 2 classes).
            pretrained (bool, optional): whether to use VGG13 encoders pretrained 
                on ImageNet for the first 8 convolutional layers in the encoder
                section of the U-Net model
            padding (int, optional): the padding applied on images at 
                convolutional layers. A padding of 1 helps keep the original
                input image's height and witdh. Note: pretrained VGG13 
                encoders use a padding of 1
        """
        super().__init__()
        if pretrained:
            encoder = models.vgg13(pretrained=True).features
            self.conv1_input = encoder[0]
            self.conv1 = encoder[2]
            self.conv2_input = encoder[5]
            self.conv2 = encoder[7]
            self.conv3_input = encoder[10]
            self.conv3 = encoder[12]
            self.conv4_input = encoder[15]
            self.conv4 = encoder[17]
        else:
            self.conv1_input = nn.Conv2d(in_channels, 64, 3, padding=padding)
            self.conv1 = nn.Conv2d(64, 64, 3, padding=padding)
            self.conv2_input = nn.Conv2d(64, 128, 3, padding=padding)
            self.conv2 = nn.Conv2d(128, 128, 3, padding=padding)
            self.conv3_input = nn.Conv2d(128, 256, 3, padding=padding)
            self.conv3 = nn.Conv2d(256, 256, 3, padding=padding)
            self.conv4_input = nn.Conv2d(256, 512, 3, padding=padding)
            self.conv4 = nn.Conv2d(512, 512, 3, padding=padding)
        self.conv5_input = nn.Conv2d(512, 1024, 3, padding=padding)
        self.conv5 = nn.Conv2d(1024, 1024, 3, padding=padding)
        self.conv6_up = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.conv6_input = nn.Conv2d(1024, 512, 3, padding=padding)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=padding)
        self.conv7_up = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv7_input = nn.Conv2d(512, 256, 3, padding=padding)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=padding)
        self.conv8_up = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv8_input = nn.Conv2d(256, 128, 3, padding=padding)
        self.conv8 = nn.Conv2d(128, 128, 3, padding=padding)
        self.conv9_up = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv9_input = nn.Conv2d(128, 64, 3, padding=padding)
        self.conv9 = nn.Conv2d(64, 64, 3, padding=padding)
        self.conv9_output = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        layer1 = F.relu(self.conv1_input(x))
        layer1 = F.relu(self.conv1(layer1))
        layer2 = F.max_pool2d(layer1, 2)
        layer2 = F.relu(self.conv2_input(layer2))
        layer2 = F.relu(self.conv2(layer2))
        layer3 = F.max_pool2d(layer2, 2)
        layer3 = F.relu(self.conv3_input(layer3))
        layer3 = F.relu(self.conv3(layer3))
        layer4 = F.max_pool2d(layer3, 2)
        layer4 = F.relu(self.conv4_input(layer4))
        layer4 = F.relu(self.conv4(layer4))
        layer5 = F.max_pool2d(layer4, 2)
        layer5 = F.relu(self.conv5_input(layer5))
        layer5 = F.relu(self.conv5(layer5))
        layer6 = F.relu(self.conv6_up(layer5))
        layer6 = torch.cat((layer4, layer6), 1)
        layer6 = F.relu(self.conv6_input(layer6))
        layer6 = F.relu(self.conv6(layer6))
        layer7 = F.relu(self.conv7_up(layer6))
        layer7 = torch.cat((layer3, layer7), 1)
        layer7 = F.relu(self.conv7_input(layer7))
        layer7 = F.relu(self.conv7(layer7))
        layer8 = F.relu(self.conv8_up(layer7))
        layer8 = torch.cat((layer2, layer8), 1)
        layer8 = F.relu(self.conv8_input(layer8))
        layer8 = F.relu(self.conv8(layer8))
        layer9 = F.relu(self.conv9_up(layer8))
        layer9 = torch.cat((layer1, layer9), 1)
        layer9 = F.relu(self.conv9_input(layer9))
        layer9 = F.relu(self.conv9(layer9))
        layer9 = self.conv9_output(layer9)
        return layer9


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
