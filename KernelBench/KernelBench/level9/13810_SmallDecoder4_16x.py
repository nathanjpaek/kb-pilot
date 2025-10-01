import torch
import torch.nn as nn


class SmallDecoder4_16x(nn.Module):

    def __init__(self, model=None, fixed=False):
        super(SmallDecoder4_16x, self).__init__()
        self.fixed = fixed
        self.conv41 = nn.Conv2d(128, 64, 3, 1, 0)
        self.conv34 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv33 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv32 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv31 = nn.Conv2d(64, 32, 3, 1, 0)
        self.conv22 = nn.Conv2d(32, 32, 3, 1, 0, dilation=1)
        self.conv21 = nn.Conv2d(32, 16, 3, 1, 0, dilation=1)
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 0, dilation=1)
        self.conv11 = nn.Conv2d(16, 3, 3, 1, 0, dilation=1)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.unpool_pwct = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        if model:
            weights = torch.load(model, map_location=lambda storage,
                location: storage)
            if 'model' in weights:
                self.load_state_dict(weights['model'])
            else:
                self.load_state_dict(weights)
            None
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, y):
        y = self.relu(self.conv41(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv34(self.pad(y)))
        y = self.relu(self.conv33(self.pad(y)))
        y = self.relu(self.conv32(self.pad(y)))
        y = self.relu(self.conv31(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv22(self.pad(y)))
        y = self.relu(self.conv21(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv12(self.pad(y)))
        y = self.relu(self.conv11(self.pad(y)))
        return y

    def forward_pwct(self, x, pool1_idx=None, pool1_size=None, pool2_idx=
        None, pool2_size=None, pool3_idx=None, pool3_size=None):
        out41 = self.relu(self.conv41(self.pad(x)))
        out41 = self.unpool_pwct(out41, pool3_idx, output_size=pool3_size)
        out34 = self.relu(self.conv34(self.pad(out41)))
        out33 = self.relu(self.conv33(self.pad(out34)))
        out32 = self.relu(self.conv32(self.pad(out33)))
        out31 = self.relu(self.conv31(self.pad(out32)))
        out31 = self.unpool_pwct(out31, pool2_idx, output_size=pool2_size)
        out22 = self.relu(self.conv22(self.pad(out31)))
        out21 = self.relu(self.conv21(self.pad(out22)))
        out21 = self.unpool_pwct(out21, pool1_idx, output_size=pool1_size)
        out12 = self.relu(self.conv12(self.pad(out21)))
        out11 = self.conv11(self.pad(out12))
        return out11


def get_inputs():
    return [torch.rand([4, 128, 4, 4])]


def get_init_inputs():
    return [[], {}]
