import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class AttentionConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        assert self.out_channels % self.groups == 0, 'out_channels should be divided by groups. (example: out_channels: 40, groups: 4)'
        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1,
            kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1,
            kernel_size), requires_grad=True)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
            bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=
            1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=
            1, bias=bias)
        self.reset_parameters()

    def forward(self, x):
        batch, _channels, height, width = x.size()
        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self
            .padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)
        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3,
            self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3,
            self.kernel_size, self.stride)
        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)
        k_out = k_out.contiguous().view(batch, self.groups, self.
            out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.
            out_channels // self.groups, height, width, -1)
        q_out = q_out.view(batch, self.groups, self.out_channels // self.
            groups, height, width, 1)
        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch,
            -1, height, width)
        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out',
            nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out',
            nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out',
            nonlinearity='relu')
        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)


class BaseClassifier(nn.Module):
    _timestep_dimension = 2
    _max_timesteps = 2048
    _value_to_pad = 0

    @staticmethod
    def pad_variable_timesteps(tensor, timestep_dimension=
        _timestep_dimension, max_timesteps=_max_timesteps, value_to_pad=
        _value_to_pad):
        """
        Pads a variable-length tensor to a fixed length along the specified dimension.
        e.g. shape [1, 1, 200, 1280] --> [1, 1, 512, 1280], with dim=2.
        
        timestep_dimension: Dimension to pad along. For shape [a, b, c, d], the respective dims are [0, 1, 2, 3].
        max_timesteps: Number of elements to pad until.
        value_to_pad: Constant value used for padding.
        """
        number_of_timesteps = tensor.size(dim=timestep_dimension)
        assert number_of_timesteps <= max_timesteps, 'Input received that is longer than . Unable to pad.'
        number_of_padded_timesteps = max_timesteps - number_of_timesteps
        padding = [0, 0] * (len(tensor.shape) - timestep_dimension - 1) + [
            0, number_of_padded_timesteps]
        return F.pad(tensor, padding, 'constant', value_to_pad)


class ANNClassifier(BaseClassifier):

    def __init__(self, input_dim):
        super(ANNClassifier, self).__init__()
        self._pool_value = 4
        self.hidden_dim_1 = hidden_dim_1 = 20
        self.linear_input_dim = hidden_dim_1 * input_dim // self._pool_value
        self.out_dim = 80
        attn_hyperparams = {'kernel_size': 3, 'padding': 1}
        self.attn = AttentionConv(1, hidden_dim_1, **attn_hyperparams)
        self.dense = nn.Linear(self.linear_input_dim, self.out_dim)
        self.pool = nn.MaxPool2d((1, self._pool_value))

    def reshape_input(self, feature, group_size):
        down_sample_len = feature.size(1) // group_size
        feature = feature[:, :down_sample_len * group_size, :]
        reshape_feature = feature.reshape(feature.size(0) * down_sample_len,
            group_size * feature.size(2))
        return reshape_feature

    def forward(self, feature):
        feature = feature.unsqueeze(dim=0).unsqueeze(dim=1)
        feature = self.pool(feature)
        feature = self.attn(feature)
        feature = F.relu(feature)
        feature = feature.transpose(0, 1)
        feature = feature.view(-1, self.linear_input_dim)
        feature = torch.stack([self.dense(feature[index].unsqueeze(dim=0)) for
            index in range(feature.size()[0])])
        feature = feature.transpose(0, 1)
        return feature


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
