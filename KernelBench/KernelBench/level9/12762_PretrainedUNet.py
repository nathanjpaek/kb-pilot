import torch
import torchvision


class Block(torch.nn.Module):

    def __init__(self, in_channels, mid_channel, out_channels, batch_norm=False
        ):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=
            mid_channel, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=mid_channel, out_channels=
            out_channels, kernel_size=3, padding=1)
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = torch.nn.BatchNorm2d(mid_channel)
            self.bn2 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = torch.nn.ReLU(inplace=True)(x)
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        out = torch.nn.ReLU(inplace=True)(x)
        return out


class PretrainedUNet(torch.nn.Module):

    def up(self, x, size):
        return torch.nn.functional.interpolate(x, size=size, mode=self.
            upscale_mode)

    def down(self, x):
        return torch.nn.MaxPool2d(kernel_size=2)(x)

    def __init__(self, in_channels, out_channels, batch_norm=False,
        upscale_mode='nearest'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.upscale_mode = upscale_mode
        self.init_conv = torch.nn.Conv2d(in_channels, 3, 1)
        endcoder = torchvision.models.vgg11(pretrained=True).features
        self.conv1 = endcoder[0]
        self.conv2 = endcoder[3]
        self.conv3 = endcoder[6]
        self.conv3s = endcoder[8]
        self.conv4 = endcoder[11]
        self.conv4s = endcoder[13]
        self.conv5 = endcoder[16]
        self.conv5s = endcoder[18]
        self.center = Block(512, 512, 256, batch_norm)
        self.dec5 = Block(512 + 256, 512, 256, batch_norm)
        self.dec4 = Block(512 + 256, 512, 128, batch_norm)
        self.dec3 = Block(256 + 128, 256, 64, batch_norm)
        self.dec2 = Block(128 + 64, 128, 32, batch_norm)
        self.dec1 = Block(64 + 32, 64, 32, batch_norm)
        self.out = torch.nn.Conv2d(in_channels=32, out_channels=
            out_channels, kernel_size=1)

    def forward(self, x):
        init_conv = torch.nn.ReLU(inplace=True)(self.init_conv(x))
        enc1 = torch.nn.ReLU(inplace=True)(self.conv1(init_conv))
        enc2 = torch.nn.ReLU(inplace=True)(self.conv2(self.down(enc1)))
        enc3 = torch.nn.ReLU(inplace=True)(self.conv3(self.down(enc2)))
        enc3 = torch.nn.ReLU(inplace=True)(self.conv3s(enc3))
        enc4 = torch.nn.ReLU(inplace=True)(self.conv4(self.down(enc3)))
        enc4 = torch.nn.ReLU(inplace=True)(self.conv4s(enc4))
        enc5 = torch.nn.ReLU(inplace=True)(self.conv5(self.down(enc4)))
        enc5 = torch.nn.ReLU(inplace=True)(self.conv5s(enc5))
        center = self.center(self.down(enc5))
        dec5 = self.dec5(torch.cat([self.up(center, enc5.size()[-2:]), enc5
            ], 1))
        dec4 = self.dec4(torch.cat([self.up(dec5, enc4.size()[-2:]), enc4], 1))
        dec3 = self.dec3(torch.cat([self.up(dec4, enc3.size()[-2:]), enc3], 1))
        dec2 = self.dec2(torch.cat([self.up(dec3, enc2.size()[-2:]), enc2], 1))
        dec1 = self.dec1(torch.cat([self.up(dec2, enc1.size()[-2:]), enc1], 1))
        out = self.out(dec1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
