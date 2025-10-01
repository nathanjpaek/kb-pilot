import torch
import torch.nn as nn


class ConvLSTMCls(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        Convolutional LSTM block for generation network

        Args:
        - in_channels: Int. Number of channels of the input of Conv2D
        - out_channels: Int. Number of channels of the result of Conv2D
        - skip_out_channels: Int. Number of channels of the result of transposed Conv2D in skip connection
        """
        super(ConvLSTMCls, self).__init__()
        in_channels = in_channels + out_channels
        self.forget_conv = nn.Conv2d(in_channels, out_channels, kernel_size
            =5, stride=1, padding=2)
        self.input_conv_1 = nn.Conv2d(in_channels, out_channels,
            kernel_size=5, stride=1, padding=2)
        self.input_conv_2 = nn.Conv2d(in_channels, out_channels,
            kernel_size=5, stride=1, padding=2)
        self.output_conv = nn.Conv2d(in_channels, out_channels, kernel_size
            =5, stride=1, padding=2)

    def forward(self, input, hidden_in, cell_in):
        """
        Forward propagation

        Args:
        - input: A tuple of Tensors.
        - hidden_in: A Tensor of shape (B, C, H, W). Hidden variable from previous ConvLSTM cell.
        - cell_in: A Tensor of shape (B, C, H, W). Cell state from previous ConvLSTM cell.

        Returns: A tuple of Tensors.
        - hidden: A Tensor of shape (B, C, H, W).
        - cell: A Tensor of shape (B, C, H, W).
        """
        x = torch.cat((input, hidden_in), dim=1)
        forget_gate = torch.sigmoid(self.forget_conv(x))
        input_gate = torch.sigmoid(self.input_conv_1(x)) * torch.tanh(self.
            input_conv_2(x))
        output_gate = torch.sigmoid(self.output_conv(x))
        cell = forget_gate * cell_in + input_gate
        hidden = output_gate * torch.tanh(cell)
        return hidden, cell


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
