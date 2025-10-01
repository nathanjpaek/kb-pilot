import torch
from torch import nn


def Binarize(tensor):
    """ 
        Binarize function: binarize input tensors
        Input:
            tensor: the input tensor. 
        Output:
            binarized: the binarized tensor.
    """
    binarized = torch.where(tensor > 0, torch.ones_like(tensor, dtype=torch
        .float32, device='cuda'), torch.full(tensor.shape, -1, dtype=torch.
        float32, device='cuda'))
    return binarized


def quantization(input, bits):
    """
        Combination of quantization and de-quantization function
        Input: 
            input: the original full-precision tensor.
            bits: number of quantized bits.
        Output:
            dequantized: the de-quantized tensor.
    """
    quantized_max = 2 ** (bits - 1) - 1
    quantized_min = -2 ** (bits - 1)
    pmax = input.max()
    pmin = input.min()
    scale_int = quantized_max - quantized_min
    scale_fp = pmax - pmin
    quantized = torch.round((input - pmin) * (scale_int / scale_fp)
        ) + quantized_min
    dequantized = (quantized - quantized_min) * (scale_fp / scale_int) + pmin
    return dequantized


class mix_Linear(nn.Module):
    """
        class mix_Linear: provide implementations of 32-bit and 1-bit layers
        Input:
            input (Tensor)
            bit_1_quantize (1-bit quantization flag)
            layer_number (the layer number)
            bit_32_quantize (full-precision flag)
        Output:
            out (Tensor)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(mix_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.quantized_weight = nn.Parameter(torch.Tensor(out_features,
            in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, bit_1_quantize=False, layer_number=0,
        bit_32_quantize=False):
        if layer_number > 2:
            if input.dtype == torch.half:
                input.data = Binarize(input.data).half()
            else:
                input.data = Binarize(input.data)
        if bit_32_quantize is True:
            if input.data.dtype == torch.half:
                out = nn.functional.linear(input, self.weight.half(), self.bias
                    )
            else:
                out = nn.functional.linear(input, self.weight, self.bias)
        elif bit_1_quantize is True:
            input.data = Binarize(input.data)
            with torch.no_grad():
                if input.data.dtype == torch.float:
                    self.quantized_weight.data = Binarize(self.weight)
                else:
                    input.data = input.data.half()
                    self.quantized_weight.data = Binarize(self.weight)
                    self.quantized_weight.data = (self.quantized_weight.
                        data.half())
            out = nn.functional.linear(input, self.quantized_weight, self.bias)
        else:
            if not hasattr(self.weight, 'org'):
                self.weight.org = self.weight.data.clone()
            self.weight.data = quantization(self.weight.org, 8)
            if input.dtype == torch.half:
                self.weight.data = self.weight.data
            out = nn.functional.linear(input, self.weight, self.bias)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
