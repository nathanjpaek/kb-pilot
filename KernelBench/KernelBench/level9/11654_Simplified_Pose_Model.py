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


class Simplified_Pose_Model(nn.Module):

    def __init__(self):
        super(Simplified_Pose_Model, self).__init__()
        no_relu_layers = ['conv5_5_CPM_L1', 'conv5_5_CPM_L2',
            'Mconv7_stage2_L1', 'Mconv7_stage2_L2', 'Mconv7_stage3_L1',
            'Mconv7_stage3_L2', 'Mconv7_stage4_L1', 'Mconv7_stage4_L2',
            'Mconv7_stage5_L1', 'Mconv7_stage5_L2', 'Mconv7_stage6_L1',
            'Mconv7_stage6_L1']
        block0 = OrderedDict({'conv1_1': [3, 64, 3, 1, 1], 'conv1_2': [64, 
            64, 3, 1, 1], 'pool1_stage1': [2, 2, 0], 'conv2_1': [64, 128, 3,
            1, 1], 'conv2_2': [128, 128, 3, 1, 1], 'pool2_stage1': [2, 2, 0
            ], 'conv3_1': [128, 256, 3, 1, 1], 'conv3_2': [256, 256, 3, 1, 
            1], 'conv3_3': [256, 256, 3, 1, 1], 'conv3_4': [256, 256, 3, 1,
            1], 'pool3_stage1': [2, 2, 0], 'conv4_1': [256, 512, 3, 1, 1],
            'conv4_2': [512, 512, 3, 1, 1], 'conv4_3_CPM': [512, 256, 3, 1,
            1], 'conv4_4_CPM': [256, 128, 3, 1, 1]})
        block1_2 = OrderedDict({'conv5_1_CPM_L2': [128, 128, 3, 1, 1],
            'conv5_2_CPM_L2': [128, 128, 3, 1, 1], 'conv5_3_CPM_L2': [128, 
            128, 3, 1, 1], 'conv5_4_CPM_L2': [128, 512, 1, 1, 0],
            'conv5_5_CPM_L2': [512, 19, 1, 1, 0]})
        self.model0 = make_layers(block0, no_relu_layers)
        self.model1_2 = make_layers(block1_2, no_relu_layers)

    def forward(self, x):
        out0 = self.model0(x)
        out1 = self.model1_2(out0)
        return out1


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
