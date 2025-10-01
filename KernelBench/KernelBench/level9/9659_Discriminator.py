import torch
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def weights_init(self):
        classname = self.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(self.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(self.weight.data, 1.0, 0.02)
            nn.init.constant_(self.bias.data, 0)

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path, map_location=self.device))
        self.eval()


class Discriminator(BaseModel):

    def __init__(self, device):
        super(Discriminator, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(3, 64, 5, 2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5, 2, 2)
        self.conv3 = nn.Conv2d(128, 256, 5, 2, 2)
        self.conv4 = nn.Conv2d(256, 512, 5, 2, 2)
        self.conv5 = nn.Conv2d(512, 1, 5, 2, 2)
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.weights_init()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.conv5(x)
        x = self.sigmoid(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {'device': 0}]
