import torch
from collections import OrderedDict
import torch.nn as nn


def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                kernel_size=v[2], stride=v[3], padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
    return nn.Sequential(OrderedDict(layers))


class handpose_model(nn.Module):

    def __init__(self):
        super(handpose_model, self).__init__()
        no_relu_layers = ['conv6_2_CPM', 'Mconv7_stage2', 'Mconv7_stage3',
            'Mconv7_stage4', 'Mconv7_stage5', 'Mconv7_stage6']
        block1_0 = OrderedDict([('conv1_1', [3, 64, 3, 1, 1]), ('conv1_2',
            [64, 64, 3, 1, 1]), ('pool1_stage1', [2, 2, 0]), ('conv2_1', [
            64, 128, 3, 1, 1]), ('conv2_2', [128, 128, 3, 1, 1]), (
            'pool2_stage1', [2, 2, 0]), ('conv3_1', [128, 256, 3, 1, 1]), (
            'conv3_2', [256, 256, 3, 1, 1]), ('conv3_3', [256, 256, 3, 1, 1
            ]), ('conv3_4', [256, 256, 3, 1, 1]), ('pool3_stage1', [2, 2, 0
            ]), ('conv4_1', [256, 512, 3, 1, 1]), ('conv4_2', [512, 512, 3,
            1, 1]), ('conv4_3', [512, 512, 3, 1, 1]), ('conv4_4', [512, 512,
            3, 1, 1]), ('conv5_1', [512, 512, 3, 1, 1]), ('conv5_2', [512, 
            512, 3, 1, 1]), ('conv5_3_CPM', [512, 128, 3, 1, 1])])
        block1_1 = OrderedDict([('conv6_1_CPM', [128, 512, 1, 1, 0]), (
            'conv6_2_CPM', [512, 22, 1, 1, 0])])
        blocks = {}
        blocks['block1_0'] = block1_0
        blocks['block1_1'] = block1_1
        for i in range(2, 7):
            blocks['block%d' % i] = OrderedDict([('Mconv1_stage%d' % i, [
                150, 128, 7, 1, 3]), ('Mconv2_stage%d' % i, [128, 128, 7, 1,
                3]), ('Mconv3_stage%d' % i, [128, 128, 7, 1, 3]), (
                'Mconv4_stage%d' % i, [128, 128, 7, 1, 3]), (
                'Mconv5_stage%d' % i, [128, 128, 7, 1, 3]), (
                'Mconv6_stage%d' % i, [128, 128, 1, 1, 0]), (
                'Mconv7_stage%d' % i, [128, 22, 1, 1, 0])])
        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)
        self.model1_0 = blocks['block1_0']
        self.model1_1 = blocks['block1_1']
        self.model2 = blocks['block2']
        self.model3 = blocks['block3']
        self.model4 = blocks['block4']
        self.model5 = blocks['block5']
        self.model6 = blocks['block6']

    def forward(self, x):
        out1_0 = self.model1_0(x)
        out1_1 = self.model1_1(out1_0)
        concat_stage2 = torch.cat([out1_1, out1_0], 1)
        out_stage2 = self.model2(concat_stage2)
        concat_stage3 = torch.cat([out_stage2, out1_0], 1)
        out_stage3 = self.model3(concat_stage3)
        concat_stage4 = torch.cat([out_stage3, out1_0], 1)
        out_stage4 = self.model4(concat_stage4)
        concat_stage5 = torch.cat([out_stage4, out1_0], 1)
        out_stage5 = self.model5(concat_stage5)
        concat_stage6 = torch.cat([out_stage5, out1_0], 1)
        out_stage6 = self.model6(concat_stage6)
        return out_stage6


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
