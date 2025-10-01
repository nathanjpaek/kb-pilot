import torch
import torch.nn as nn
from torch.nn import init


class VGG_block(nn.Module):
    """ 1. default have the bias
        2. using ReLU and 3 * max pooling
        3. 10 layers of VGG original
        4. 2 extra layers by CMU
        5. default in_dim = 3,out_dim = 128
        6. all kernal_size = 3, stride = 1
    """

    def __init__(self, in_dim=3, out_dim=128):
        super(VGG_block, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool_1 = nn.MaxPool2d(2, 2, 0)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool_2 = nn.MaxPool2d(2, 2, 0)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_4 = nn.ReLU(inplace=True)
        self.pool_3 = nn.MaxPool2d(2, 2, 0)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_2 = nn.PReLU(num_parameters=512)
        self.conv4_3_cmu = nn.Conv2d(512, 256, 3, 1, 1)
        self.relu4_3 = nn.PReLU(num_parameters=256)
        self.conv4_4_cmu = nn.Conv2d(256, 128, 3, 1, 1)
        self.relu4_4 = nn.PReLU(num_parameters=128)
        self.initilization()

    def forward(self, input_1):
        """inplace middle result """
        output_1 = self.conv1_1(input_1)
        output_1 = self.relu1_1(output_1)
        output_1 = self.conv1_2(output_1)
        output_1 = self.relu1_2(output_1)
        output_1 = self.pool_1(output_1)
        output_1 = self.conv2_1(output_1)
        output_1 = self.relu2_1(output_1)
        output_1 = self.conv2_2(output_1)
        output_1 = self.relu2_2(output_1)
        output_1 = self.pool_2(output_1)
        output_1 = self.conv3_1(output_1)
        output_1 = self.relu3_1(output_1)
        output_1 = self.conv3_2(output_1)
        output_1 = self.relu3_2(output_1)
        output_1 = self.conv3_3(output_1)
        output_1 = self.relu3_3(output_1)
        output_1 = self.conv3_4(output_1)
        output_1 = self.relu3_4(output_1)
        output_1 = self.pool_3(output_1)
        output_1 = self.conv4_1(output_1)
        output_1 = self.relu4_1(output_1)
        output_1 = self.conv4_2(output_1)
        output_1 = self.relu4_2(output_1)
        output_1 = self.conv4_3_cmu(output_1)
        output_1 = self.relu4_3(output_1)
        output_1 = self.conv4_4_cmu(output_1)
        output_1 = self.relu4_4(output_1)
        return output_1

    def initilization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
            else:
                try:
                    init.constant_(m.weight, 0.0)
                except:
                    pass


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
