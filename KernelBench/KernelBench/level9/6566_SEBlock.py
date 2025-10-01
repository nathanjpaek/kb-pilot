import torch
import torch.nn.functional as F
import torch.nn


def to_device(device_object, tensor):
    """
    Select device for non-parameters tensor w.r.t model or tensor which has been specified a device.
    """
    if isinstance(device_object, torch.nn.Module):
        next(device_object.parameters()).device
    elif isinstance(device_object, torch.Tensor):
        pass
    return tensor


class TdnnAffine(torch.nn.Module):
    """ An implemented tdnn affine component by conv1d
        y = splice(w * x, context) + b

    @input_dim: number of dims of frame <=> inputs channels of conv
    @output_dim: number of layer nodes <=> outputs channels of conv
    @context: a list of context
        e.g.  [-2,0,2]
    If context is [0], then the TdnnAffine is equal to linear layer.
    """

    def __init__(self, input_dim, output_dim, context=[0], bias=True, pad=
        True, norm_w=False, norm_f=False, subsampling_factor=1):
        super(TdnnAffine, self).__init__()
        for index in range(0, len(context) - 1):
            if context[index] >= context[index + 1]:
                raise ValueError(
                    'Context tuple {} is invalid, such as the order.'.
                    format(context))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context = context
        self.bool_bias = bias
        self.pad = pad
        self.norm_w = norm_w
        self.norm_f = norm_f
        self.stride = subsampling_factor
        self.left_context = context[0] if context[0] < 0 else 0
        self.right_context = context[-1] if context[-1] > 0 else 0
        self.tot_context = self.right_context - self.left_context + 1
        if self.tot_context > 1 and self.norm_f:
            self.norm_f = False
            None
        kernel_size = self.tot_context,
        self.weight = torch.nn.Parameter(torch.randn(output_dim, input_dim,
            *kernel_size))
        if self.bool_bias:
            self.bias = torch.nn.Parameter(torch.randn(output_dim))
        else:
            self.register_parameter('bias', None)
        self.init_weight()
        if len(context) != self.tot_context:
            self.mask = torch.tensor([[[(1 if index in context else 0) for
                index in range(self.left_context, self.right_context + 1)]]])
        else:
            self.mask = None
        self.selected_device = False

    def init_weight(self):
        torch.nn.init.normal_(self.weight, 0.0, 0.01)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim
        if self.pad:
            inputs = F.pad(inputs, (-self.left_context, self.right_context),
                mode='constant', value=0)
        assert inputs.shape[2] >= self.tot_context
        if not self.selected_device and self.mask is not None:
            self.mask = to_device(self, self.mask)
            self.selected_device = True
        filters = (self.weight * self.mask if self.mask is not None else
            self.weight)
        if self.norm_w:
            filters = F.normalize(filters, dim=1)
        if self.norm_f:
            inputs = F.normalize(inputs, dim=1)
        outputs = F.conv1d(inputs, filters, self.bias, self.stride, padding
            =0, dilation=1, groups=1)
        return outputs

    def extra_repr(self):
        return (
            '{input_dim}, {output_dim}, context={context}, bias={bool_bias}, stride={stride}, pad={pad}, norm_w={norm_w}, norm_f={norm_f}'
            .format(**self.__dict__))

    @classmethod
    def thop_count(self, m, x, y):
        x = x[0]
        kernel_ops = torch.zeros(m.weight.size()[2:]).numel()
        bias_ops = 1 if m.bias is not None else 0
        total_ops = y.nelement() * (m.input_dim * kernel_ops + bias_ops)
        m.total_ops += torch.DoubleTensor([int(total_ops)])


class StatisticsPooling(torch.nn.Module):
    """ An usual mean [+ stddev] poolling layer"""

    def __init__(self, input_dim, stddev=True, unbiased=False, eps=1e-10):
        super(StatisticsPooling, self).__init__()
        self.stddev = stddev
        self.input_dim = input_dim
        if self.stddev:
            self.output_dim = 2 * input_dim
        else:
            self.output_dim = input_dim
        self.eps = eps
        self.unbiased = unbiased

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim
        counts = inputs.shape[2]
        mean = torch.unsqueeze(inputs.sum(dim=2) / counts, dim=2)
        if self.stddev:
            if self.unbiased and counts > 1:
                counts = counts - 1
            var = torch.sum((inputs - mean) ** 2, dim=2) / counts
            std = torch.unsqueeze(torch.sqrt(var.clamp(min=self.eps)), dim=2)
            return torch.cat((mean, std), dim=1)
        else:
            return mean

    def get_output_dim(self):
        return self.output_dim

    def extra_repr(self):
        return (
            '{input_dim}, {output_dim}, stddev={stddev}, unbiased={unbiased}, eps={eps}'
            .format(**self.__dict__))

    @classmethod
    def thop_count(self, m, x, y):
        pass


class SEBlock(torch.nn.Module):
    """ A SE Block layer layer which can learn to use global information to selectively emphasise informative 
    features and suppress less useful ones.
    This is a pytorch implementation of SE Block based on the paper:
    Squeeze-and-Excitation Networks
    by JFChou xmuspeech 2019-07-13
       Snowdar xmuspeech 2020-04-28 [Check and update]
    """

    def __init__(self, input_dim, ratio=4, inplace=True):
        """
        @ratio: a reduction ratio which allows us to vary the capacity and computational cost of the SE blocks 
        in the network.
        """
        super(SEBlock, self).__init__()
        self.input_dim = input_dim
        self.pooling = StatisticsPooling(input_dim, stddev=False)
        self.fc_1 = TdnnAffine(input_dim, input_dim // ratio)
        self.relu = torch.nn.ReLU(inplace=inplace)
        self.fc_2 = TdnnAffine(input_dim // ratio, input_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim
        x = self.pooling(inputs)
        x = self.relu(self.fc_1(x))
        scale = self.sigmoid(self.fc_2(x))
        return inputs * scale


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
