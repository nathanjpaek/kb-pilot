import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.linear2 = nn.Linear(512, 256)
        self.lrelu3 = nn.LeakyReLU(0.2)
        self.linear3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        out = self.linear1(img.view(img.shape[0], 784))
        out = self.linear2(self.lrelu2(out))
        out = self.linear3(self.lrelu3(out))
        out = self.sigmoid(out)
        return out


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
