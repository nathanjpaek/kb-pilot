import torch
import torch.cuda
from torch import nn


def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target + -1e+30 * (1 - mask)


class OutputLayer(nn.Module):

    def __init__(self, hidden_size):
        super(OutputLayer, self).__init__()
        self.weight1 = torch.empty(hidden_size * 2, 1)
        self.weight2 = torch.empty(hidden_size * 2, 1)
        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)
        self.weight1 = nn.Parameter(self.weight1.squeeze(), requires_grad=True)
        self.weight2 = nn.Parameter(self.weight2.squeeze(), requires_grad=True)

    def forward(self, stacked_model_output1, stacked_model_output2,
        stacked_model_output3, cmask):
        start = torch.cat((stacked_model_output1, stacked_model_output2), dim=1
            )
        end = torch.cat((stacked_model_output1, stacked_model_output3), dim=1)
        start = torch.matmul(self.weight1, start)
        end = torch.matmul(self.weight2, end)
        start = mask_logits(start, cmask)
        end = mask_logits(end, cmask)
        return start, end


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4,
        4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
