import torch
import torch.nn


class LDEPooling(torch.nn.Module):
    """A novel learnable dictionary encoding layer according to [Weicheng Cai, etc., "A NOVEL LEARNABLE 
    DICTIONARY ENCODING LAYER FOR END-TO-END LANGUAGE IDENTIFICATION", icassp, 2018]"""

    def __init__(self, input_dim, c_num=64):
        super(LDEPooling, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim * c_num
        self.mu = torch.nn.Parameter(torch.randn(input_dim, c_num))
        self.s = torch.nn.Parameter(torch.ones(c_num))
        self.softmax_for_w = torch.nn.Softmax(dim=3)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim
        r = inputs.transpose(1, 2).unsqueeze(3) - self.mu
        w = self.softmax_for_w(self.s * torch.sum(r ** 2, dim=2, keepdim=True))
        e = torch.mean(w * r, dim=1)
        return e.reshape(-1, self.output_dim, 1)

    def get_output_dim(self):
        return self.output_dim


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
