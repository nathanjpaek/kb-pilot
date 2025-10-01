import torch
import torch.nn as nn


def adder2d_function(X, W, stride=1, padding=0, groups=1):
    n_filters, _d_filter, h_filter, w_filter = W.size()
    n_x, _d_x, h_x, w_x = X.size()
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1
    h_out, w_out = int(h_out), int(w_out)
    if groups == 1:
        X_col = torch.nn.functional.unfold(X.view(1, -1, h_x, w_x),
            h_filter, dilation=1, padding=padding, stride=stride).view(n_x,
            -1, h_out * w_out)
        X_col = X_col.permute(1, 2, 0).contiguous().view(X_col.size(1), -1)
        W_col = W.view(n_filters, -1)
        out_2 = -torch.cdist(W_col, X_col.transpose(0, 1).contiguous(), 1)
        out_2 = out_2.detach()
        out = -torch.cdist(W_col, X_col.transpose(0, 1).contiguous(), 2)
        out.data = out_2.data
        out = out.view(n_filters, h_out, w_out, n_x)
        out = out.permute(3, 0, 1, 2).contiguous()
        return out
    else:
        torch.Tensor(n_x, n_filters, h_out, w_out)
        outs = []
        for g in range(groups):
            X_part = X[:, g, ...].unsqueeze(1)
            W_part = W[g, ...].unsqueeze(0)
            X_col = torch.nn.functional.unfold(X_part.view(1, -1, h_x, w_x),
                h_filter, dilation=1, padding=padding, stride=stride).view(n_x,
                -1, h_out * w_out)
            X_col = X_col.permute(1, 2, 0).contiguous().view(X_col.size(1), -1)
            W_col = W_part.view(1, -1)
            out_2 = -torch.cdist(W_col, X_col.transpose(0, 1).contiguous(), 1)
            out_2 = out_2.detach()
            out = -torch.cdist(W_col, X_col.transpose(0, 1).contiguous(), 2)
            out.data = out_2.data
            out = out.view(1, h_out, w_out, n_x)
            out = out.permute(3, 0, 1, 2).contiguous()
            outs.append(out)
        return torch.cat(outs, dim=1)


class adder2d(nn.Module):

    def __init__(self, input_channel, output_channel, kernel_size, stride=1,
        padding=0, groups=1, bias=False):
        super(adder2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.groups = groups
        self.adder = torch.nn.Parameter(nn.init.kaiming_normal_(torch.randn
            (output_channel, input_channel // groups, kernel_size,
            kernel_size)))
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(nn.init.kaiming_uniform_(torch.
                zeros(output_channel)))

    def forward(self, x):
        output = adder2d_function(x, self.adder, self.stride, self.padding,
            self.groups)
        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return output

    def extra_repr(self):
        return (
            f"{'v2'.upper()}, {self.input_channel}, {self.output_channel}, " +
            f'kenrel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, groups={self.groups}, bias={self.bias}'
            )


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channel': 4, 'output_channel': 4, 'kernel_size': 4}]
