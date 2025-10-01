import torch
import torch.nn as nn
from itertools import chain as chain
import torch.utils.data
from collections import OrderedDict
import torch.hub
import torch.nn.parallel
import torch.optim


class concatLayer(nn.Module):

    def __init__(self, in_channels, out_channels_perSub, i, j, appendix):
        super(concatLayer, self).__init__()
        self.firstSub = self.concatLayerSub(in_channels,
            out_channels_perSub, '%d_stage%d_' % (i, j) + appendix + '_0')
        self.secondSub = self.concatLayerSub(out_channels_perSub,
            out_channels_perSub, '%d_stage%d_' % (i, j) + appendix + '_1')
        self.thirdSub = self.concatLayerSub(out_channels_perSub,
            out_channels_perSub, '%d_stage%d_' % (i, j) + appendix + '_2')

    def forward(self, x):
        firstSub = self.firstSub(x)
        secondSub = self.secondSub(firstSub)
        thirdSub = self.thirdSub(secondSub)
        out = torch.cat([firstSub, secondSub, thirdSub], 1)
        return out

    def concatLayerSub(self, in_channels, out_channels, layerName):
        concatLayerSubOrdered = OrderedDict()
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        concatLayerSubOrdered.update({('Mconv' + layerName): conv2d})
        concatLayerSubOrdered.update({('Mprelu' + layerName): nn.PReLU(
            out_channels)})
        return nn.Sequential(concatLayerSubOrdered)


class stage(nn.Module):

    def __init__(self, stageID, in_channels, out_channels_perSub,
        mid_channels, out_channels, appendix):
        super(stage, self).__init__()
        self.firstConcat = concatLayer(in_channels, out_channels_perSub, 1,
            stageID, appendix)
        self.secondConcat = concatLayer(3 * out_channels_perSub,
            out_channels_perSub, 2, stageID, appendix)
        self.thirdConcat = concatLayer(3 * out_channels_perSub,
            out_channels_perSub, 3, stageID, appendix)
        self.fourthConcat = concatLayer(3 * out_channels_perSub,
            out_channels_perSub, 4, stageID, appendix)
        self.fifthConcat = concatLayer(3 * out_channels_perSub,
            out_channels_perSub, 5, stageID, appendix)
        conv2d = nn.Conv2d(3 * out_channels_perSub, mid_channels,
            kernel_size=1, padding=0)
        prelu = nn.PReLU(mid_channels)
        self.afterConcatsFirst = nn.Sequential(OrderedDict({(
            'Mconv6_stage%d_%s' % (stageID, appendix)): conv2d, (
            'Mprelu6_stage%d_%s' % (stageID, appendix)): prelu}))
        conv2d = nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0
            )
        self.afterConcatsSecond = nn.Sequential(OrderedDict({(
            'Mconv7_stage%d_%s' % (stageID, appendix)): conv2d}))

    def forward(self, x):
        x = self.firstConcat(x)
        x = self.secondConcat(x)
        x = self.thirdConcat(x)
        x = self.fourthConcat(x)
        x = self.fifthConcat(x)
        x = self.afterConcatsFirst(x)
        out = self.afterConcatsSecond(x)
        return out


class L2Part(nn.Module):

    def __init__(self, in_channels, stage_out_channels):
        super(L2Part, self).__init__()
        self.firstStage = stage(0, in_channels, 96, in_channels * 2,
            stage_out_channels, 'L2')
        self.secondStage = stage(1, in_channels + stage_out_channels,
            in_channels, in_channels * 4, stage_out_channels, 'L2')
        self.thirdStage = stage(2, in_channels + stage_out_channels,
            in_channels, in_channels * 4, stage_out_channels, 'L2')
        self.fourthStage = stage(3, in_channels + stage_out_channels,
            in_channels, in_channels * 4, stage_out_channels, 'L2')

    def forward(self, features):
        x = self.firstStage(features)
        x = torch.cat([features, x], 1)
        x = self.secondStage(x)
        x = torch.cat([features, x], 1)
        x = self.thirdStage(x)
        x = torch.cat([features, x], 1)
        out = self.fourthStage(x)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'stage_out_channels': 4}]
