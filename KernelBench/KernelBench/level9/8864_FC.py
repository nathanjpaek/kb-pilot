import torch
import torch.nn
import torch.utils.checkpoint
import torch.utils.data
import torch.optim
import torch.distributed
import torch.multiprocessing


class FC(torch.nn.Module):

    def __init__(self, in_features, out_features, act=torch.nn.ReLU(inplace
        =True)):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=True)
        self.act = act

    def forward(self, input):
        output = self.linear(input)
        output = self.act(output)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
