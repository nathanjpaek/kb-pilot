import math
import torch


def cheb_conv(laplacian, inputs, weight):
    """Chebyshev convolution.

    Args:
        laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
        inputs (:obj:`torch.Tensor`): The current input data being forwarded.
        weight (:obj:`torch.Tensor`): The weights of the current layer.

    Returns:
        :obj:`torch.Tensor`: Inputs after applying Chebyshev convolution.
    """
    B, V, Fin = inputs.shape
    K, Fin, Fout = weight.shape
    x0 = inputs.permute(1, 2, 0).contiguous()
    x0 = x0.view([V, Fin * B])
    inputs = x0.unsqueeze(0)
    if K > 0:
        x1 = torch.sparse.mm(laplacian, x0)
        inputs = torch.cat((inputs, x1.unsqueeze(0)), 0)
        for _ in range(1, K - 1):
            x2 = 2 * torch.sparse.mm(laplacian, x1) - x0
            inputs = torch.cat((inputs, x2.unsqueeze(0)), 0)
            x0, x1 = x1, x2
    inputs = inputs.view([K, V, Fin, B])
    inputs = inputs.permute(3, 1, 2, 0).contiguous()
    inputs = inputs.view([B * V, Fin * K])
    weight = weight.view(Fin * K, Fout)
    inputs = inputs.matmul(weight)
    inputs = inputs.view([B, V, Fout])
    return inputs


class ChebConv(torch.nn.Module):
    """Graph convolutional layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True,
        conv=cheb_conv):
        """Initialize the Chebyshev layer.

        Args:
            in_channels (int): Number of channels/features in the input graph.
            out_channels (int): Number of channels/features in the output graph.
            kernel_size (int): Number of trainable parameters per filter, which is also the size of the convolutional kernel.
                                The order of the Chebyshev polynomials is kernel_size - 1.
            bias (bool): Whether to add a bias term.
            conv (callable): Function which will perform the actual convolution.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._conv = conv
        shape = kernel_size, in_channels, out_channels
        self.weight = torch.nn.Parameter(torch.Tensor(*shape))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.kaiming_initialization()

    def kaiming_initialization(self):
        """Initialize weights and bias.
        """
        std = math.sqrt(2 / (self.in_channels * self.kernel_size))
        self.weight.data.normal_(0, std)
        if self.bias is not None:
            self.bias.data.fill_(0.01)

    def forward(self, laplacian, inputs):
        """Forward graph convolution.

        Args:
            laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
            inputs (:obj:`torch.Tensor`): The current input data being forwarded.

        Returns:
            :obj:`torch.Tensor`: The convoluted inputs.
        """
        outputs = self._conv(laplacian, inputs, self.weight)
        if self.bias is not None:
            outputs += self.bias
        return outputs


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
