import torch
import torch.utils.data


class EncModel(torch.nn.Module):

    def __init__(self, num_input, num_hid, num_out):
        super(EncModel, self).__init__()
        self.in_hid = torch.nn.Linear(num_input, num_hid)
        self.hid_out = torch.nn.Linear(num_hid, num_out)

    def forward(self, input):
        hid_sum = self.in_hid(input)
        hidden = torch.tanh(hid_sum)
        out_sum = self.hid_out(hidden)
        output = torch.sigmoid(out_sum)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_input': 4, 'num_hid': 4, 'num_out': 4}]
