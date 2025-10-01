import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLayer(nn.Module):

    def __init__(self, input_size, in_channels, out_channels, kernel_width,
        act_fun=nn.ReLU, drop_prob=0.1):
        """Initilize CNN layer.

        Args:
            input_size [int]: embedding dim or the last dim of the input
            in_channels [int]: number of channels for inputs
            out_channels [int]: number of channels for outputs
            kernel_width [int]: the width on sequence for the first dim of kernel
            act_fun [torch.nn.modules.activation]: activation function
            drop_prob [float]: drop out ratio
        """
        super(CNNLayer, self).__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_width = kernel_width
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_width,
            input_size))
        self.drop_out = nn.Dropout(drop_prob)
        assert callable(act_fun), TypeError(
            "Type error of 'act_fun', use functions like nn.ReLU/nn.Tanh.")
        self.act_fun = act_fun()

    def forward(self, inputs, mask=None, out_type='max'):
        """Forward propagation.

        Args:
            inputs [tensor]: input tensor (batch_size * in_channels * max_seq_len * input_size)
                             or (batch_size * max_seq_len * input_size)
            mask [tensor]: mask matrix (batch_size * max_seq_len)
            out_type [str]: use 'max'/'mean'/'all' to choose

        Returns:
            outputs [tensor]: output tensor (batch_size * out_channels) or (batch_size * left_len * n_hidden)
        """
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(1).repeat(1, self.in_channels, 1, 1)
        assert inputs.dim() == 4 and inputs.size(1
            ) == self.in_channels, "Dimension error of 'inputs'."
        assert inputs.size(-1
            ) == self.input_size, "Dimension error of 'inputs'."
        now_batch_size, _, max_seq_len, _ = inputs.size()
        assert max_seq_len >= self.kernel_width, "Dimension error of 'inputs'."
        assert out_type in ['max', 'mean', 'all'], ValueError(
            "Value error of 'out_type', only accepts 'max'/'mean'/'all'.")
        left_len = max_seq_len - self.kernel_width + 1
        if mask is None:
            mask = torch.ones((now_batch_size, left_len), device=inputs.device)
        assert mask.dim() == 2, "Dimension error of 'mask'."
        mask = mask[:, -left_len:].unsqueeze(1)
        outputs = self.conv(inputs)
        outputs = self.drop_out(outputs)
        outputs = outputs.reshape(-1, self.out_channels, left_len)
        outputs = self.act_fun(outputs)
        if out_type == 'max':
            outputs = outputs.masked_fill(~mask.bool(), float('-inf'))
            outputs = F.max_pool1d(outputs, left_len).reshape(-1, self.
                out_channels)
            outputs = outputs.masked_fill(torch.isinf(outputs), 0)
        elif out_type == 'mean':
            outputs = outputs.masked_fill(~mask.bool(), 0)
            lens = torch.sum(mask, dim=-1)
            outputs = torch.sum(outputs, dim=-1) / (lens.float() + 1e-09)
        elif out_type == 'all':
            outputs = outputs.masked_fill(~mask.bool(), 0)
            outputs = outputs.transpose(1, 2)
        return outputs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'in_channels': 4, 'out_channels': 4,
        'kernel_width': 4}]
