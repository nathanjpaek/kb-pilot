import torch
import torch.nn as nn


class CBDNet(nn.Module):

    def __init__(self):
        super(CBDNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.E01 = nn.Conv2d(3, 32, kernel_size=[3, 3], stride=(1, 1),
            padding=(1, 1))
        self.E02 = nn.Conv2d(32, 32, kernel_size=[3, 3], stride=(1, 1),
            padding=(1, 1))
        self.E03 = nn.Conv2d(32, 32, kernel_size=[3, 3], stride=(1, 1),
            padding=(1, 1))
        self.E04 = nn.Conv2d(32, 32, kernel_size=[3, 3], stride=(1, 1),
            padding=(1, 1))
        self.E05 = nn.Conv2d(32, 3, kernel_size=[3, 3], stride=(1, 1),
            padding=(1, 1))
        self.DS01_layer00 = nn.Conv2d(6, 64, kernel_size=[3, 3], stride=(1,
            1), padding=(1, 1))
        self.DS01_layer01 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1,
            1), padding=(1, 1))
        self.DS01_layer02 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1,
            1), padding=(1, 1))
        self.DS01_layer03 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1,
            1), padding=(1, 1))
        self.DS02 = nn.Conv2d(64, 256, kernel_size=[2, 2], stride=(2, 2))
        self.DS02_layer00_cf = nn.Conv2d(256, 128, kernel_size=[1, 1],
            stride=(1, 1))
        self.DS02_layer00 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=
            (1, 1), padding=(1, 1))
        self.DS02_layer01 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=
            (1, 1), padding=(1, 1))
        self.DS02_layer02 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=
            (1, 1), padding=(1, 1))
        self.DS03 = nn.Conv2d(128, 512, kernel_size=[2, 2], stride=(2, 2))
        self.DS03_layer00_cf = nn.Conv2d(512, 256, kernel_size=[1, 1],
            stride=(1, 1))
        self.DS03_layer00 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=
            (1, 1), padding=(1, 1))
        self.DS03_layer01 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=
            (1, 1), padding=(1, 1))
        self.DS03_layer02 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=
            (1, 1), padding=(1, 1))
        self.UPS03_layer00 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride
            =(1, 1), padding=(1, 1))
        self.UPS03_layer01 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride
            =(1, 1), padding=(1, 1))
        self.UPS03_layer02 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride
            =(1, 1), padding=(1, 1))
        self.UPS03_layer03 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride
            =(1, 1), padding=(1, 1))
        self.USP02 = nn.ConvTranspose2d(512, 128, kernel_size=[2, 2],
            stride=(2, 2), bias=False)
        self.US02_layer00 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=
            (1, 1), padding=(1, 1))
        self.US02_layer01 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=
            (1, 1), padding=(1, 1))
        self.US02_layer02 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=
            (1, 1), padding=(1, 1))
        self.USP01 = nn.ConvTranspose2d(256, 64, kernel_size=[2, 2], stride
            =(2, 2), bias=False)
        self.US01_layer00 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1,
            1), padding=(1, 1))
        self.US01_layer01 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1,
            1), padding=(1, 1))
        self.US01_layer02 = nn.Conv2d(64, 3, kernel_size=[3, 3], stride=(1,
            1), padding=(1, 1))

    def forward(self, input):
        x = self.E01(input)
        self.relu(x)
        x = self.E02(x)
        self.relu(x)
        x = self.E03(x)
        self.relu(x)
        x = self.E04(x)
        self.relu(x)
        x = self.E05(x)
        self.relu(x)
        noise_level = x
        x = torch.cat((input, noise_level), dim=1)
        x = self.DS01_layer00(x)
        self.relu(x)
        x = self.DS01_layer01(x)
        self.relu(x)
        x = self.DS01_layer02(x)
        self.relu(x)
        x = self.DS01_layer03(x)
        self.relu(x)
        down1_result = x
        x = self.DS02(down1_result)
        x = self.DS02_layer00_cf(x)
        x = self.DS02_layer00(x)
        self.relu(x)
        x = self.DS02_layer01(x)
        self.relu(x)
        x = self.DS02_layer02(x)
        self.relu(x)
        down2_result = x
        x = self.DS03(down2_result)
        x = self.DS03_layer00_cf(x)
        x = self.DS03_layer00(x)
        self.relu(x)
        x = self.DS03_layer01(x)
        self.relu(x)
        x = self.DS03_layer02(x)
        self.relu(x)
        x = self.UPS03_layer00(x)
        self.relu(x)
        x = self.UPS03_layer01(x)
        self.relu(x)
        x = self.UPS03_layer02(x)
        self.relu(x)
        x = self.UPS03_layer03(x)
        self.relu(x)
        x = self.USP02(x)
        x = torch.add(x, 1, down2_result)
        del down2_result
        x = self.US02_layer00(x)
        self.relu(x)
        x = self.US02_layer01(x)
        self.relu(x)
        x = self.US02_layer02(x)
        self.relu(x)
        x = self.USP01(x)
        x = torch.add(x, 1, down1_result)
        del down1_result
        x = self.US01_layer00(x)
        self.relu(x)
        x = self.US01_layer01(x)
        self.relu(x)
        x = self.US01_layer02(x)
        y = torch.add(input, 1, x)
        del x
        return noise_level, y


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
