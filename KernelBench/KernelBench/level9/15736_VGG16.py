import torch
import numpy as np
import torchvision.transforms.functional as F
import torch.nn as nn
import torch.nn.functional as F


class Normalize:

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def undo(self, imgarr):
        proc_img = imgarr.copy()
        proc_img[..., 0] = (self.std[0] * imgarr[..., 0] + self.mean[0]
            ) * 255.0
        proc_img[..., 1] = (self.std[1] * imgarr[..., 1] + self.mean[1]
            ) * 255.0
        proc_img[..., 2] = (self.std[2] * imgarr[..., 2] + self.mean[2]
            ) * 255.0
        return proc_img

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)
        proc_img[..., 0] = (imgarr[..., 0] / 255.0 - self.mean[0]) / self.std[0
            ]
        proc_img[..., 1] = (imgarr[..., 1] / 255.0 - self.mean[1]) / self.std[1
            ]
        proc_img[..., 2] = (imgarr[..., 2] / 255.0 - self.mean[2]) / self.std[2
            ]
        return proc_img


class BaseNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.normalize = Normalize()
        self.NormLayer = nn.BatchNorm2d
        self.not_training = []
        self.bn_frozen = []
        self.from_scratch_layers = []

    def _init_weights(self, path_to_weights):
        None
        weights_dict = torch.load(path_to_weights)
        self.load_state_dict(weights_dict, strict=False)

    def fan_out(self):
        raise NotImplementedError

    def fixed_layers(self):
        return self.not_training

    def _fix_running_stats(self, layer, fix_params=False):
        if isinstance(layer, self.NormLayer):
            self.bn_frozen.append(layer)
            if fix_params and layer not in self.not_training:
                self.not_training.append(layer)
        elif isinstance(layer, list):
            for m in layer:
                self._fix_running_stats(m, fix_params)
        else:
            for m in layer.children():
                self._fix_running_stats(m, fix_params)

    def _fix_params(self, layer):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, self.NormLayer
            ) or isinstance(layer, nn.Linear):
            self.not_training.append(layer)
            if isinstance(layer, self.NormLayer):
                self.bn_frozen.append(layer)
        elif isinstance(layer, list):
            for m in layer:
                self._fix_params(m)
        elif isinstance(layer, nn.Module):
            if hasattr(layer, 'weight') or hasattr(layer, 'bias'):
                None
            for m in layer.children():
                self._fix_params(m)

    def _freeze_bn(self, layer):
        if isinstance(layer, self.NormLayer):
            layer.eval()
        elif isinstance(layer, nn.Module):
            for m in layer.children():
                self._freeze_bn(m)

    def train(self, mode=True):
        super().train(mode)
        for layer in self.not_training:
            if hasattr(layer, 'weight') and layer.weight is not None:
                layer.weight.requires_grad = False
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.requires_grad = False
            elif isinstance(layer, torch.nn.Module):
                None
        for bn_layer in self.bn_frozen:
            self._freeze_bn(bn_layer)

    def _lr_mult(self):
        return 1.0, 2.0, 10.0, 20

    def parameter_groups(self, base_lr, wd):
        w_old, b_old, w_new, b_new = self._lr_mult()
        groups = {'params': [], 'weight_decay': wd, 'lr': w_old * base_lr}, {
            'params': [], 'weight_decay': 0.0, 'lr': b_old * base_lr}, {
            'params': [], 'weight_decay': wd, 'lr': w_new * base_lr}, {'params'
            : [], 'weight_decay': 0.0, 'lr': b_new * base_lr}
        fixed_layers = self.fixed_layers()
        for m in self.modules():
            if m in fixed_layers:
                continue
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear
                ) or isinstance(m, self.NormLayer):
                if m.weight is not None:
                    if m in self.from_scratch_layers:
                        groups[2]['params'].append(m.weight)
                    else:
                        groups[0]['params'].append(m.weight)
                if m.bias is not None:
                    if m in self.from_scratch_layers:
                        groups[3]['params'].append(m.bias)
                    else:
                        groups[1]['params'].append(m.bias)
            elif hasattr(m, 'weight'):
                None
        for i, g in enumerate(groups):
            None
        return groups


class VGG16(BaseNet):

    def __init__(self, fc6_dilation=1):
        super(VGG16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.fc6 = nn.Conv2d(512, 1024, 3, padding=fc6_dilation, dilation=
            fc6_dilation)
        self.drop6 = nn.Dropout2d(p=0.5)
        self.fc7 = nn.Conv2d(1024, 1024, 1)
        self._fix_params([self.conv1_1, self.conv1_2])

    def fan_out(self):
        return 1024

    def forward(self, x):
        return self.forward_as_dict(x)['conv6']

    def forward_as_dict(self, x):
        x = F.relu(self.conv1_1(x), inplace=True)
        x = F.relu(self.conv1_2(x), inplace=True)
        x = self.pool1(x)
        x = F.relu(self.conv2_1(x), inplace=True)
        x = F.relu(self.conv2_2(x), inplace=True)
        x = self.pool2(x)
        x = F.relu(self.conv3_1(x), inplace=True)
        x = F.relu(self.conv3_2(x), inplace=True)
        x = F.relu(self.conv3_3(x), inplace=True)
        conv3 = x
        x = self.pool3(x)
        x = F.relu(self.conv4_1(x), inplace=True)
        x = F.relu(self.conv4_2(x), inplace=True)
        x = F.relu(self.conv4_3(x), inplace=True)
        x = self.pool4(x)
        x = F.relu(self.conv5_1(x), inplace=True)
        x = F.relu(self.conv5_2(x), inplace=True)
        x = F.relu(self.conv5_3(x), inplace=True)
        x = F.relu(self.fc6(x), inplace=True)
        x = self.drop6(x)
        x = F.relu(self.fc7(x), inplace=True)
        conv6 = x
        return dict({'conv3': conv3, 'conv6': conv6})


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
