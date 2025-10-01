import copy
import torch
import torch.nn as nn
import torch.utils.data
from torchvision.models.squeezenet import squeezenet1_0
from torchvision.models.squeezenet import squeezenet1_1
import torch.nn.modules.activation


class GramMatrix(nn.Module):

    def forward(self, x):
        b, c, h, w = x.size()
        F = x.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h * w)
        return G


class GramDiag(nn.Module):
    """
    docstring for GramDiag
    """

    def __init__(self, gram_diagonal_squared=False):
        super().__init__()
        self.__gram_diagonal_squared = gram_diagonal_squared

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, 1, h * w)
        gram_diag = None
        for b in range(x.size(0)):
            if self.__gram_diagonal_squared:
                z = torch.bmm(x[b] * x[b], (x[b] * x[b]).transpose(2, 1))
            else:
                z = torch.bmm(x[b], x[b].transpose(2, 1))
            if isinstance(gram_diag, torch.Tensor):
                gram_diag = torch.cat(gram_diag, z)
            else:
                gram_diag = z
        gram_diag = torch.squeeze(gram_diag).unsqueeze(0)
        return gram_diag.div_(h * w)


class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000, pretrained=False,
        layer='', gram=False, gram_diag=False, gram_diagonal_squared=False):
        super().__init__()
        if version not in [1.0, 1.1]:
            raise ValueError(
                'Unsupported SqueezeNet version {version}:1.0 or 1.1 expected'
                .format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            pytorch_squeeze = squeezenet1_0(pretrained, num_classes=num_classes
                )
            features_names = ['conv_1', 'relu_1', 'maxpool_1', 'fire_2',
                'fire_3', 'fire_4', 'maxpool_4', 'fire_5', 'fire_6',
                'fire_7', 'fire_8', 'maxpool_8', 'fire_9']
        else:
            pytorch_squeeze = squeezenet1_1(pretrained, num_classes=num_classes
                )
            features_names = ['conv_1', 'relu_1', 'maxpool_1', 'fire_2',
                'fire_3', 'maxpool_3', 'fire_4', 'fire_5', 'maxpool_5',
                'fire_6', 'fire_7', 'fire_8', 'fire_9']
        classifier_names = ['drop_10', 'conv_10', 'relu_10', 'avgpool_10']
        self.features = torch.nn.Sequential()
        for name, module in zip(features_names, pytorch_squeeze.features):
            self.features.add_module(name, copy.deepcopy(module))
            if layer is name:
                break
        if len(features_names) == len(self.features
            ) and layer != features_names[-1]:
            for name, module in zip(classifier_names, pytorch_squeeze.
                classifier):
                self.features.add_module(name, copy.deepcopy(module))
                if layer is name:
                    break
        del pytorch_squeeze
        if gram:
            self.features.add_module('gram matrix', GramMatrix())
        elif gram_diag:
            self.features.add_module('gram diagonal', GramDiag(
                gram_diagonal_squared))

    def forward(self, x):
        return self.features(x)


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
