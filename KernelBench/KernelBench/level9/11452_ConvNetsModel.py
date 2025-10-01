import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNetsModel(nn.Module):

    def __init__(self, num_classes, cross_entropy_loss=False, kernel_size=3,
        channel_size1=32, channel_size2=64, dropout=False):
        super(ConvNetsModel, self).__init__()
        self.cross_entropy_loss = cross_entropy_loss
        self.kernel_size = kernel_size
        self.w_dropout = dropout
        self.channel_size1 = channel_size1
        self.channel_size2 = channel_size2
        self.cn1 = nn.Conv2d(in_channels=3, out_channels=channel_size1,
            kernel_size=(kernel_size, kernel_size))
        self.cn2 = nn.Conv2d(in_channels=channel_size1, out_channels=
            channel_size1, kernel_size=(kernel_size, kernel_size))
        self.pool = nn.MaxPool2d((2, 2))
        if dropout:
            self.dropout = nn.Dropout(p=0.3)
            self.dropout2 = nn.Dropout(p=0.6)
        self.cn3 = nn.Conv2d(in_channels=channel_size1, out_channels=
            channel_size2, kernel_size=(kernel_size, kernel_size))
        self.cn4 = nn.Conv2d(in_channels=channel_size2, out_channels=
            channel_size2, kernel_size=(kernel_size, kernel_size))
        self.new_image_dim = ((32 + 2 * (-kernel_size + 1)) // 2 + 2 * (-
            kernel_size + 1)) // 2
        self.fc1 = nn.Linear(channel_size2 * self.new_image_dim * self.
            new_image_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)
        if not self.cross_entropy_loss:
            self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.cn1(x)
        x = F.relu(x)
        x = self.cn2(x)
        x = self.pool(x)
        if self.w_dropout:
            x = self.dropout(x)
        x = F.relu(x)
        x = self.cn3(x)
        x = F.relu(x)
        x = self.cn4(x)
        x = self.pool(x)
        if self.w_dropout:
            x = self.dropout(x)
        x = F.relu(x)
        x = x.view(-1, self.channel_size2 * self.new_image_dim * self.
            new_image_dim)
        x = self.fc1(x)
        if self.w_dropout:
            x = self.dropout2(x)
        x = F.relu(x)
        x = self.fc2(x)
        if not self.cross_entropy_loss:
            x = self.softmax(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 32, 32])]


def get_init_inputs():
    return [[], {'num_classes': 4}]
