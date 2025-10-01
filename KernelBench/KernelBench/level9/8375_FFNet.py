import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNet(nn.Module):
    """Modified ResidualNetworkSegment model class"""

    def __init__(self, block, num_blocks, width, depth):
        super(FFNet, self).__init__()
        assert (depth - 4
            ) % 4 == 0, 'Depth not compatible with recurrent architectue.'
        self.iters = (depth - 4) // 4
        self.in_planes = int(width)
        self.conv1 = nn.Conv1d(1, width, kernel_size=3, stride=1, padding=1,
            bias=False)
        layers = []
        for _ in range(self.iters):
            for i in range(len(num_blocks)):
                layers.append(self._make_layer(block, width, num_blocks[i],
                    stride=1))
        self.recur_block = nn.Sequential(*layers)
        self.conv2 = nn.Conv1d(width, width, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.conv3 = nn.Conv1d(width, int(width / 2), kernel_size=3, stride
            =1, padding=1, bias=False)
        self.conv4 = nn.Conv1d(int(width / 2), 2, kernel_size=3, stride=1,
            padding=1, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for strd in strides:
            layers.append(block(self.in_planes, planes, strd))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.recur_block(out)
        thought = F.relu(self.conv2(out))
        thought = F.relu(self.conv3(thought))
        thought = self.conv4(thought)
        return thought


def get_inputs():
    return [torch.rand([4, 1, 64])]


def get_init_inputs():
    return [[], {'block': 4, 'num_blocks': 4, 'width': 4, 'depth': 4}]
