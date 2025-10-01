import torch
import torch.nn as nn
import torch.nn.functional as F


class LayoutNet(nn.Module):

    def __init__(self):
        super(LayoutNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=1)
        self.conv7 = nn.Conv2d(1024, 2048, kernel_size=3, padding=1, stride=1)
        self.deconv00 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1,
            stride=1)
        self.deconv0 = nn.Conv2d(1024 * 2, 512, kernel_size=3, padding=1,
            stride=1)
        self.deconv1 = nn.Conv2d(512 * 2, 256, kernel_size=3, padding=1,
            stride=1)
        self.deconv2 = nn.Conv2d(256 * 2, 128, kernel_size=3, padding=1,
            stride=1)
        self.deconv3 = nn.Conv2d(128 * 2, 64, kernel_size=3, padding=1,
            stride=1)
        self.deconv4 = nn.Conv2d(64 * 2, 32, kernel_size=3, padding=1, stride=1
            )
        self.deconv5 = nn.Conv2d(32 * 2, 3, kernel_size=3, padding=1, stride=1)
        self.deconv6_sf = nn.Sigmoid()
        self.deconv00_c = nn.Conv2d(2048, 1024, kernel_size=3, padding=1,
            stride=1)
        self.deconv0_c = nn.Conv2d(1024 * 3, 512, kernel_size=3, padding=1,
            stride=1)
        self.deconv1_c = nn.Conv2d(512 * 3, 256, kernel_size=3, padding=1,
            stride=1)
        self.deconv2_c = nn.Conv2d(256 * 3, 128, kernel_size=3, padding=1,
            stride=1)
        self.deconv3_c = nn.Conv2d(128 * 3, 64, kernel_size=3, padding=1,
            stride=1)
        self.deconv4_c = nn.Conv2d(64 * 3, 32, kernel_size=3, padding=1,
            stride=1)
        self.deconv5_c = nn.Conv2d(32 * 3, 16, kernel_size=3, padding=1,
            stride=1)
        self.deconv6_sf_c = nn.Sigmoid()
        self.ref1 = nn.Linear(2048 * 4 * 4, 1024)
        self.ref2 = nn.Linear(1024, 256)
        self.ref3 = nn.Linear(256, 64)
        self.ref4 = nn.Linear(64, 11)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1_relu = self.relu(conv1)
        pool1 = self.pool(conv1_relu)
        conv2 = self.conv2(pool1)
        conv2_relu = self.relu(conv2)
        pool2 = self.pool(conv2_relu)
        conv3 = self.conv3(pool2)
        conv3_relu = self.relu(conv3)
        pool3 = self.pool(conv3_relu)
        conv4 = self.conv4(pool3)
        conv4_relu = self.relu(conv4)
        pool4 = self.pool(conv4_relu)
        conv5 = self.conv5(pool4)
        conv5_relu = self.relu(conv5)
        pool5 = self.pool(conv5_relu)
        conv6 = self.conv6(pool5)
        conv6_relu = self.relu(conv6)
        pool6 = self.pool(conv6_relu)
        conv7 = self.conv7(pool6)
        conv7_relu = self.relu(conv7)
        pool7 = self.pool(conv7_relu)
        unpool00 = F.interpolate(pool7, scale_factor=2)
        deconv00 = self.deconv00(unpool00)
        deconv00_relu = self.relu(deconv00)
        unpool0_ = torch.cat((deconv00_relu, pool6), dim=1)
        unpool0 = F.interpolate(unpool0_, scale_factor=2)
        deconv0 = self.deconv0(unpool0)
        deconv0_relu = self.relu(deconv0)
        unpool1_ = torch.cat((deconv0_relu, pool5), dim=1)
        unpool1 = F.interpolate(unpool1_, scale_factor=2)
        deconv1 = self.deconv1(unpool1)
        deconv1_relu = self.relu(deconv1)
        unpool2_ = torch.cat((deconv1_relu, pool4), dim=1)
        unpool2 = F.interpolate(unpool2_, scale_factor=2)
        deconv2 = self.deconv2(unpool2)
        deconv2_relu = self.relu(deconv2)
        unpool3_ = torch.cat((deconv2_relu, pool3), dim=1)
        unpool3 = F.interpolate(unpool3_, scale_factor=2)
        deconv3 = self.deconv3(unpool3)
        deconv3_relu = self.relu(deconv3)
        unpool4_ = torch.cat((deconv3_relu, pool2), dim=1)
        unpool4 = F.interpolate(unpool4_, scale_factor=2)
        deconv4 = self.deconv4(unpool4)
        deconv4_relu = self.relu(deconv4)
        unpool5_ = torch.cat((deconv4_relu, pool1), dim=1)
        unpool5 = F.interpolate(unpool5_, scale_factor=2)
        deconv5 = self.deconv5(unpool5)
        deconv6_sf = self.deconv6_sf(deconv5)
        deconv00_c = self.deconv00_c(unpool00)
        deconv00_relu_c = self.relu(deconv00_c)
        unpool0_c = torch.cat((deconv00_relu_c, unpool0_), dim=1)
        unpool0_c = F.interpolate(unpool0_c, scale_factor=2)
        deconv0_c = self.deconv0_c(unpool0_c)
        deconv0_relu_c = self.relu(deconv0_c)
        unpool1_c = torch.cat((deconv0_relu_c, unpool1_), dim=1)
        unpool1_c = F.interpolate(unpool1_c, scale_factor=2)
        deconv1_c = self.deconv1_c(unpool1_c)
        deconv1_relu_c = self.relu(deconv1_c)
        unpool2_c = torch.cat((deconv1_relu_c, unpool2_), dim=1)
        unpool2_c = F.interpolate(unpool2_c, scale_factor=2)
        deconv2_c = self.deconv2_c(unpool2_c)
        deconv2_relu_c = self.relu(deconv2_c)
        unpool3_c = torch.cat((deconv2_relu_c, unpool3_), dim=1)
        unpool3_c = F.interpolate(unpool3_c, scale_factor=2)
        deconv3_c = self.deconv3_c(unpool3_c)
        deconv3_relu_c = self.relu(deconv3_c)
        unpool4_c = torch.cat((deconv3_relu_c, unpool4_), dim=1)
        unpool4_c = F.interpolate(unpool4_c, scale_factor=2)
        deconv4_c = self.deconv4_c(unpool4_c)
        deconv4_relu_c = self.relu(deconv4_c)
        unpool5_c = torch.cat((deconv4_relu_c, unpool5_), dim=1)
        unpool5_c = F.interpolate(unpool5_c, scale_factor=2)
        deconv5_c = self.deconv5_c(unpool5_c)
        deconv6_sf_c = self.deconv6_sf_c(deconv5_c)
        ref0 = pool7.view(-1, 2048 * 4 * 4)
        ref1 = self.ref1(ref0)
        ref1_relu = self.relu(ref1)
        ref2 = self.ref2(ref1_relu)
        ref2_relu = self.relu(ref2)
        ref3 = self.ref3(ref2_relu)
        ref3_relu = self.relu(ref3)
        ref4 = self.ref4(ref3_relu)
        return deconv6_sf, deconv6_sf_c, ref4


def get_inputs():
    return [torch.rand([4, 3, 256, 256])]


def get_init_inputs():
    return [[], {}]
