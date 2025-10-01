import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class EnDeWithPooling(nn.Module):

    def __init__(self, activation, initType, numChannels, batchnorm=False,
        softmax=False):
        super(EnDeWithPooling, self).__init__()
        self.batchnorm = batchnorm
        self.bias = not batchnorm
        self.initType = initType
        self.activation = None
        self.numChannels = numChannels
        self.softmax = softmax
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.SELU(inplace=True)
        self.conv1 = nn.Conv2d(self.numChannels, 16, 3, 1, 1, bias=self.bias)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1, bias=self.bias)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1, bias=self.bias)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, 2, 1, 1)
        self.deconv1 = nn.ConvTranspose2d(16, 8, 3, 2, 1, 1)
        self.classifier = nn.Conv2d(8, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.intermediate = nn.Conv2d(64, 64, 1, 1, 0, bias=self.bias)
        self.skip1 = nn.Conv2d(16, 16, 1, 1, 0, bias=self.bias)
        self.skip2 = nn.Conv2d(32, 32, 1, 1, 0, bias=self.bias)
        if self.batchnorm:
            self.bn1 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn3 = nn.BatchNorm2d(64)
            self.bn4 = nn.BatchNorm2d(32)
            self.bn5 = nn.BatchNorm2d(16)
            self.bn6 = nn.BatchNorm2d(8)

    def forward(self, x):
        if self.batchnorm:
            conv1_ = self.pool(self.bn1(self.activation(self.conv1(x))))
            conv2_ = self.pool(self.bn2(self.activation(self.conv2(conv1_))))
            conv3_ = self.pool(self.bn3(self.activation(self.conv3(conv2_))))
            intermediate_ = self.activation(self.intermediate(conv3_))
            skip_deconv3_ = self.deconv3(intermediate_) + self.activation(self
                .skip2(conv2_))
            deconv3_ = self.bn4(self.activation(skip_deconv3_))
            skip_deconv2_ = self.deconv2(deconv3_) + self.activation(self.
                skip1(conv1_))
            deconv2_ = self.bn5(self.activation(skip_deconv2_))
            deconv1_ = self.bn6(self.activation(self.deconv1(deconv2_)))
            score = self.classifier(deconv1_)
        else:
            conv1_ = self.pool(self.activation(self.conv1(x)))
            conv2_ = self.pool(self.activation(self.conv2(conv1_)))
            conv3_ = self.pool(self.activation(self.conv3(conv2_)))
            intermediate_ = self.activation(self.intermediate(conv3_))
            skip_deconv3_ = self.deconv3(intermediate_) + self.activation(self
                .skip2(conv2_))
            deconv3_ = self.activation(skip_deconv3_)
            skip_deconv2_ = self.deconv2(deconv3_) + self.activation(self.
                skip1(conv1_))
            deconv2_ = self.activation(skip_deconv2_)
            deconv1_ = self.activation(self.deconv1(deconv2_))
            if self.softmax:
                score = F.softmax(self.classifier(deconv1_), dim=1)
            else:
                score = self.classifier(deconv1_)
        return score

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.initType == 'default':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, np.sqrt(2.0 / n))
                elif self.initType == 'xavier':
                    nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                if self.initType == 'default':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, np.sqrt(2.0 / n))
                elif self.initType == 'xavier':
                    nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'activation': 4, 'initType': 4, 'numChannels': 4}]
