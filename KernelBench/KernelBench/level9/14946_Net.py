import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter


class Conv1dExt(nn.Conv1d):

    def __init__(self, *args, **kwargs):
        super(Conv1dExt, self).__init__(*args, **kwargs)
        self.init_ncc()
        self.input_tied_modules = []
        self.output_tied_modules = []

    def init_ncc(self):
        w = self.weight.view(self.weight.size(0), -1)
        mean = torch.mean(w, dim=1).unsqueeze(1)
        self.t0_factor = w - mean
        self.t0_norm = torch.norm(w, p=2, dim=1)
        self.start_ncc = Variable(torch.zeros(self.out_channels))
        self.start_ncc = self.normalized_cross_correlation()

    def normalized_cross_correlation(self):
        w = self.weight.view(self.weight.size(0), -1)
        t_norm = torch.norm(w, p=2, dim=1)
        if self.in_channels == 1 & sum(self.kernel_size) == 1:
            ncc = w.squeeze() / torch.norm(self.t0_norm, p=2)
            ncc = ncc - self.start_ncc
            return ncc
        mean = torch.mean(w, dim=1).unsqueeze(1)
        t_factor = w - mean
        h_product = self.t0_factor * t_factor
        cov = torch.sum(h_product, dim=1)
        denom = self.t0_norm * t_norm
        ncc = cov / denom
        ncc = ncc - self.start_ncc
        return ncc

    def split_output_channel(self, channel_i):
        """Split one output channel (a feature) into two, but retain summed value

            Args:
                channel_i: (int) number of channel to be split.  the ith channel
        """
        self.out_channels += 1
        orig_weight = self.weight.data
        split_pos = 2 * torch.rand(self.in_channels, self.kernel_size[0])
        new_weight = torch.zeros(self.out_channels, self.in_channels, self.
            kernel_size[0])
        if channel_i > 0:
            new_weight[:channel_i, :, :] = orig_weight[:channel_i, :, :]
        new_weight[channel_i, :, :] = orig_weight[channel_i, :, :] * split_pos
        new_weight[channel_i + 1, :, :] = orig_weight[channel_i, :, :] * (2 -
            split_pos)
        if channel_i + 2 < self.out_channels:
            new_weight[channel_i + 2, :, :] = orig_weight[channel_i + 1, :, :]
        if self.bias is not None:
            orig_bias = self.bias.data
            new_bias = torch.zeros(self.out_channels)
            new_bias[:channel_i + 1] = orig_bias[:channel_i + 1]
            new_bias[channel_i + 1:] = orig_bias[channel_i:]
            self.bias = Parameter(new_bias)
        self.weight = Parameter(new_weight)
        self.init_ncc()

    def split_input_channel(self, channel_i):
        if channel_i > self.in_channels:
            None
            return
        self.in_channels += 1
        orig_weight = self.weight.data
        dup_slice = orig_weight[:, channel_i, :] * 0.5
        new_weight = torch.zeros(self.out_channels, self.in_channels, self.
            kernel_size[0])
        if channel_i > 0:
            new_weight[:, :channel_i, :] = orig_weight[:, :channel_i, :]
        new_weight[:, channel_i, :] = dup_slice
        new_weight[:, channel_i + 1, :] = dup_slice
        if channel_i + 1 < self.in_channels:
            new_weight[:, channel_i + 2, :] = orig_weight[:, channel_i + 1, :]
        self.weight = Parameter(new_weight)
        self.init_ncc()

    def split_feature(self, feature_i):
        """Splits feature in output and input channels

            Args:
                feature_i: (int)
        """
        self.split_output_channel(channel_i=feature_i)
        for dep in self.input_tied_modules:
            dep.split_input_channel(channel_i=feature_i)
        for dep in self.output_tied_modules:
            dep.split_output_channel(channel_i=feature_i)

    def split_features(self, threshold):
        """Decides which features to split if they are below a specific threshold

            Args:
                threshold: (float?) less than 1.
        """
        ncc = self.normalized_cross_correlation()
        for i, ncc_val in enumerate(ncc):
            if ncc_val < threshold:
                None
                self.split_feature(i)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv1dExt(in_channels=1, out_channels=4, kernel_size=1,
            bias=False)
        self.conv2 = Conv1dExt(in_channels=1, out_channels=4, kernel_size=1,
            bias=False)
        self.conv3 = Conv1dExt(in_channels=4, out_channels=4, kernel_size=1,
            bias=False)
        self.conv4 = Conv1dExt(in_channels=4, out_channels=2, kernel_size=1,
            bias=True)
        self.conv1.input_tied_modules = [self.conv3]
        self.conv1.output_tied_modules = [self.conv2]
        self.conv2.input_tied_modules = [self.conv3]
        self.conv2.output_tied_modules = [self.conv1]
        self.conv3.input_tied_modules = [self.conv4]

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = nn.functional.relu(x1 + x2)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        return x


def get_inputs():
    return [torch.rand([4, 1, 64])]


def get_init_inputs():
    return [[], {}]
