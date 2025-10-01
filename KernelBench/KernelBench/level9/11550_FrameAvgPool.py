from _paritybench_helpers import _mock_config
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn


class FrameAvgPool(nn.Module):

    def __init__(self, cfg):
        super(FrameAvgPool, self).__init__()
        input_size = cfg.INPUT_SIZE
        hidden_size = cfg.HIDDEN_SIZE
        kernel_size = cfg.KERNEL_SIZE
        stride = cfg.STRIDE
        self.vis_conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.avg_pool = nn.AvgPool1d(kernel_size, stride)

    def forward(self, visual_input):
        vis_h = torch.relu(self.vis_conv(visual_input))
        vis_h = self.avg_pool(vis_h)
        return vis_h


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'cfg': _mock_config(INPUT_SIZE=4, HIDDEN_SIZE=4,
        KERNEL_SIZE=4, STRIDE=4)}]
