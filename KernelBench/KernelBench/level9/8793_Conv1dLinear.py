import torch
import torch.multiprocessing
import torch.nn
import torch.optim
import torch.distributed


class Conv1dLinear(torch.nn.Module):
    """Conv1D + Linear for Transformer block.

    A variant of MultiLayeredConv1d, which replaces second conv-layer to linear.

    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        """Initialize Conv1dLinear module.

        Args:
            in_chans (int): Number of input channels.
            hidden_chans (int): Number of hidden channels.
            kernel_size (int): Kernel size of conv1d.
            dropout_rate (float): Dropout rate.

        """
        super(Conv1dLinear, self).__init__()
        self.w_1 = torch.nn.Conv1d(in_chans, hidden_chans, kernel_size,
            stride=1, padding=(kernel_size - 1) // 2)
        self.w_2 = torch.nn.Linear(hidden_chans, in_chans)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (torch.Tensor): Batch of input tensors (B, T, in_chans).

        Returns:
            torch.Tensor: Batch of output tensors (B, T, hidden_chans).

        """
        x = torch.relu(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(self.dropout(x))


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_chans': 4, 'hidden_chans': 4, 'kernel_size': 4,
        'dropout_rate': 0.5}]
