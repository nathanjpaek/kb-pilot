import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        keep_rate = 0.5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=
            3, stride=1, padding='same', bias=True)
        self.dropout1 = nn.Dropout2d(1 - keep_rate)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size
            =3, stride=1, padding='same', bias=True)
        self.maxpooling1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size
            =3, stride=1, padding='same', bias=True)
        self.dropout2 = nn.Dropout2d(1 - keep_rate)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size
            =3, stride=1, padding='same', bias=True)
        self.maxpooling2 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size
            =3, stride=1, padding='same', bias=True)
        self.dropout3 = nn.Dropout2d(1 - keep_rate)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size
            =3, stride=1, padding='same', bias=True)
        self.maxpooling3 = nn.MaxPool2d(2)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=128,
            kernel_size=3, stride=1, padding='same', bias=True)
        self.dropout4 = nn.Dropout2d(1 - keep_rate)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128,
            kernel_size=3, stride=1, padding='same', bias=True)
        self.maxpooling4 = nn.MaxPool2d(2)
        self.conv9 = nn.Conv2d(in_channels=128, out_channels=256,
            kernel_size=3, stride=1, padding='same', bias=True)
        self.dropout5 = nn.Dropout2d(1 - keep_rate)
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=3, stride=1, padding='same', bias=True)
        self.conv11 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
            kernel_size=2, stride=2, padding=0, bias=True)
        self.conv12 = nn.Conv2d(in_channels=256, out_channels=128,
            kernel_size=3, stride=1, padding='same', bias=True)
        self.dropout6 = nn.Dropout2d(1 - keep_rate)
        self.conv13 = nn.Conv2d(in_channels=128, out_channels=128,
            kernel_size=3, stride=1, padding='same', bias=True)
        self.conv14 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
            kernel_size=2, stride=2, padding=0, bias=True)
        self.conv15 = nn.Conv2d(in_channels=128, out_channels=64,
            kernel_size=3, stride=1, padding='same', bias=True)
        self.dropout7 = nn.Dropout2d(1 - keep_rate)
        self.conv16 = nn.Conv2d(in_channels=64, out_channels=64,
            kernel_size=3, stride=1, padding='same', bias=True)
        self.conv17 = nn.ConvTranspose2d(in_channels=64, out_channels=32,
            kernel_size=2, stride=2, padding=0, bias=True)
        self.conv18 = nn.Conv2d(in_channels=64, out_channels=32,
            kernel_size=3, stride=1, padding='same', bias=True)
        self.dropout8 = nn.Dropout2d(1 - keep_rate)
        self.conv19 = nn.Conv2d(in_channels=32, out_channels=32,
            kernel_size=3, stride=1, padding='same', bias=True)
        self.conv20 = nn.ConvTranspose2d(in_channels=32, out_channels=16,
            kernel_size=2, stride=2, padding=0, bias=True)
        self.conv21 = nn.Conv2d(in_channels=32, out_channels=16,
            kernel_size=3, stride=1, padding='same', bias=True)
        self.dropout9 = nn.Dropout2d(1 - keep_rate)
        self.conv22 = nn.Conv2d(in_channels=16, out_channels=16,
            kernel_size=3, stride=1, padding='same', bias=True)
        self.outputs = nn.Conv2d(in_channels=16, out_channels=1,
            kernel_size=1, stride=1, padding='same', bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x1 = self.maxpooling1(x)
        x1 = self.conv3(x1)
        x1 = F.relu(x1)
        x1 = self.dropout2(x1)
        x1 = self.conv4(x1)
        x1 = F.relu(x1)
        x2 = self.maxpooling2(x1)
        x2 = self.conv5(x2)
        x2 = F.relu(x2)
        x2 = self.dropout3(x2)
        x2 = self.conv6(x2)
        x2 = F.relu(x2)
        x3 = self.maxpooling3(x2)
        x3 = self.conv7(x3)
        x3 = F.relu(x3)
        x3 = self.dropout4(x3)
        x3 = self.conv8(x3)
        x3 = F.relu(x3)
        x4 = self.maxpooling4(x3)
        x4 = self.conv9(x4)
        x4 = F.relu(x4)
        x4 = self.dropout5(x4)
        x4 = self.conv10(x4)
        x4 = F.relu(x4)
        x5 = self.conv11(x4)
        x5 = torch.cat((x5, x3), 1)
        x5 = self.conv12(x5)
        x5 = F.relu(x5)
        x5 = self.dropout6(x5)
        x5 = self.conv13(x5)
        x5 = F.relu(x5)
        x6 = self.conv14(x5)
        x6 = torch.cat((x6, x2), 1)
        x6 = self.conv15(x6)
        x6 = F.relu(x6)
        x6 = self.dropout7(x6)
        x6 = self.conv16(x6)
        x6 = F.relu(x6)
        x7 = self.conv17(x6)
        x7 = torch.cat((x7, x1), 1)
        x7 = self.conv18(x7)
        x7 = F.relu(x7)
        x7 = self.dropout8(x7)
        x7 = self.conv19(x7)
        x7 = F.relu(x7)
        x8 = self.conv20(x7)
        x8 = torch.cat((x8, x), 1)
        x8 = self.conv21(x8)
        x8 = F.relu(x8)
        x8 = self.dropout9(x8)
        x8 = self.conv22(x8)
        x8 = F.relu(x8)
        outputs = self.outputs(x8)
        out = torch.sigmoid(outputs)
        return out


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
