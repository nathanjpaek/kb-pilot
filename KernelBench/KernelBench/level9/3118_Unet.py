import torch
from torch import nn
from torch.nn import functional as F


class ContractingBlock(nn.Module):
    """
    ContractingBlock Class
    Performs two convolutions followed by a max pool operation.
    Values:
        input_channels: the number of channels to expect from a given input
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3x3_0 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.conv3x3_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3)

    def forward(self, x):
        """
        Function for completing a forward pass of ContractingBlock: 
        Given an image tensor, completes a contracting block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        """
        fx = F.relu(self.conv3x3_0(x))
        fx = F.relu(self.conv3x3_1(fx))
        return fx


class ExpandingBlock(nn.Module):
    """
    ExpandingBlock
    Performs an upsampling, a convolution, a concatenation of its two inputs,
    followed by two more convolutions.
    Values:
        input_channels: the number of channels to expect from a given input
    """

    def __init__(self, hid_channels):
        super(ExpandingBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
            align_corners=True)
        self.conv = nn.Conv2d(hid_channels, hid_channels // 2, kernel_size=
            3, stride=1, padding=1)
        self.conv3x3_0 = nn.Conv2d(hid_channels, hid_channels // 2,
            kernel_size=3, stride=1)
        self.conv3x3_1 = nn.Conv2d(hid_channels // 2, hid_channels // 2,
            kernel_size=3, stride=1)

    @staticmethod
    def crop(x, shape=None):
        """
        Function for cropping an image tensor: Given an image tensor and the new shape,
        crops to the center pixels (assumes that the input's size and the new size are
        even numbers).
        Parameters:
            image: image tensor of shape (batch size, channels, height, width)
            new_shape: a torch.Size object with the shape you want x to have
        """
        _, _, h, w = x.shape
        _, _, h_new, w_new = shape
        ch, cw = h // 2, w // 2
        ch_new, cw_new = h_new // 2, w_new // 2
        x1 = int(cw - cw_new)
        y1 = int(ch - ch_new)
        x2 = int(x1 + w_new)
        y2 = int(y1 + h_new)
        return x[:, :, y1:y2, x1:x2]

    def forward(self, x, skip):
        """
        Function for completing a forward pass of ExpandingBlock: 
        Given an image tensor, completes an expanding block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
            skip_con_x: the image tensor from the contracting path (from the opposing block of x)
                    for the skip connection
        """
        up = self.upsample(x)
        upconv = self.conv(up)
        skip = self.crop(skip, upconv.shape)
        fx = torch.cat([upconv, skip], dim=1)
        fx = F.relu(self.conv3x3_0(fx))
        fx = F.relu(self.conv3x3_1(fx))
        return fx


class FeatureMapBlock(nn.Module):
    """
    FeatureMapBlock
    The final layer of a UNet - 
    maps each pixel to a pixel with the correct number of output dimensions
    using a 1x1 convolution.
    Values:
        input_channels: the number of channels to expect from a given input
    """

    def __init__(self, in_channels, out_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Function for completing a forward pass of FeatureMapBlock: 
        Given an image tensor, returns it mapped to the desired number of channels.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        """
        fx = self.conv1x1(x)
        return fx


class Unet(nn.Module):
    """
    UNet Class
    A series of 4 contracting blocks followed by 4 expanding blocks to 
    transform an input image into the corresponding paired image, with an upfeature
    layer at the start and a downfeature layer at the end
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    """

    def __init__(self, in_channels=1, hid_channels=64, out_channels=1):
        super(Unet, self).__init__()
        self.contract1 = ContractingBlock(1, hid_channels)
        self.contract2 = ContractingBlock(hid_channels, hid_channels * 2)
        self.contract3 = ContractingBlock(hid_channels * 2, hid_channels * 4)
        self.contract4 = ContractingBlock(hid_channels * 4, hid_channels * 8)
        self.bottleneck = ContractingBlock(hid_channels * 8, hid_channels * 16)
        self.expand1 = ExpandingBlock(hid_channels * 16)
        self.expand2 = ExpandingBlock(hid_channels * 8)
        self.expand3 = ExpandingBlock(hid_channels * 4)
        self.expand4 = ExpandingBlock(hid_channels * 2)
        self.downfeature = FeatureMapBlock(hid_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Function for completing a forward pass of UNet: 
        Given an image tensor, passes it through U-Net and returns the output.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        """
        d1 = self.contract1(x)
        dp1 = self.pool(d1)
        d2 = self.contract2(dp1)
        dp2 = self.pool(d2)
        d3 = self.contract3(dp2)
        dp3 = self.pool(d3)
        d4 = self.contract4(dp3)
        dp4 = self.pool(d4)
        b = self.bottleneck(dp4)
        up1 = self.expand1(b, d4)
        up2 = self.expand2(up1, d3)
        up3 = self.expand3(up2, d2)
        up4 = self.expand4(up3, d1)
        xn = self.downfeature(up4)
        return xn


def get_inputs():
    return [torch.rand([4, 1, 256, 256])]


def get_init_inputs():
    return [[], {}]
