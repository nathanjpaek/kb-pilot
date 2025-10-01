import torch
import torch.nn.parallel
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn


class FrameMaxPool(nn.Module):

    def __init__(self, input_size, hidden_size, stride):
        super(FrameMaxPool, self).__init__()
        self.vis_conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.max_pool = nn.MaxPool1d(stride)

    def forward(self, visual_input):
        vis_h = torch.relu(self.vis_conv(visual_input))
        vis_h = self.max_pool(vis_h)
        return vis_h


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'stride': 1}]
