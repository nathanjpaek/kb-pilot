import torch
import torch.nn as nn
import torch.utils.data.dataloader


class SegmentationHead(nn.Module):

    def __init__(self, descriptor_dimension, num_classes, **kwargs):
        super().__init__()
        self.descriptor_dimension = descriptor_dimension
        self.classifier = nn.Conv2d(in_channels=descriptor_dimension,
            out_channels=num_classes, kernel_size=1, bias=True)

    def forward(self, input):
        return self.classifier(input[0].detach())


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'descriptor_dimension': 4, 'num_classes': 4}]
