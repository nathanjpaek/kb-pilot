import torch


class CSNet(torch.nn.Module):

    def __init__(self):
        super(CSNet, self).__init__()
        k_stride = 20
        color_channel = 3
        mr = 12
        self.conv0 = torch.nn.Conv2d(in_channels=color_channel,
            out_channels=mr, kernel_size=2 * k_stride, stride=k_stride,
            padding=k_stride)
        self.deconv0 = torch.nn.ConvTranspose2d(in_channels=mr,
            out_channels=color_channel, kernel_size=2 * k_stride, stride=
            k_stride, padding=k_stride)
        self.conv1_1 = torch.nn.Conv2d(in_channels=color_channel,
            out_channels=64, kernel_size=11, stride=1, padding=5)
        self.conv1_2 = torch.nn.Conv2d(in_channels=64, out_channels=32,
            kernel_size=1, stride=1, padding=0)
        self.conv1_3 = torch.nn.Conv2d(in_channels=32, out_channels=
            color_channel, kernel_size=7, stride=1, padding=3)
        self.conv2_1 = torch.nn.Conv2d(in_channels=color_channel,
            out_channels=64, kernel_size=11, stride=1, padding=5)
        self.conv2_2 = torch.nn.Conv2d(in_channels=64, out_channels=32,
            kernel_size=1, stride=1, padding=0)
        self.conv2_3 = torch.nn.Conv2d(in_channels=32, out_channels=
            color_channel, kernel_size=7, stride=1, padding=3)
        self.conv3_1 = torch.nn.Conv2d(in_channels=color_channel,
            out_channels=64, kernel_size=11, stride=1, padding=5)
        self.conv3_2 = torch.nn.Conv2d(in_channels=64, out_channels=32,
            kernel_size=1, stride=1, padding=0)
        self.conv3_3 = torch.nn.Conv2d(in_channels=32, out_channels=
            color_channel, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        measurement = self.conv0(x)
        y0 = self.deconv0(measurement)
        y = torch.nn.functional.relu(self.conv1_1(y0))
        y = torch.nn.functional.relu(self.conv1_2(y))
        y1 = y0 + self.conv1_3(y)
        y = torch.nn.functional.relu(self.conv2_1(y1))
        y = torch.nn.functional.relu(self.conv2_2(y))
        y2 = y1 + self.conv2_3(y)
        y = torch.nn.functional.relu(self.conv3_1(y2))
        y = torch.nn.functional.relu(self.conv3_2(y))
        y = y2 + self.conv3_3(y)
        return measurement, y


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
