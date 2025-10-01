import torch
import torch.utils.data


class DNNnet(torch.nn.Module):

    def __init__(self, n_layer, n_in_channel, n_out_channel):
        super(DNNnet, self).__init__()
        self.n_layer = n_layer
        self.fc_layers = torch.nn.ModuleList()
        self.act_func = torch.nn.Sigmoid()
        start_layer = torch.nn.Linear(n_in_channel, 2048)
        self.start = start_layer
        for i in range(self.n_layer - 1):
            fc_layer = torch.nn.Linear(2048, 2048)
            self.fc_layers.append(fc_layer)
        end_layer = torch.nn.Linear(2048, n_out_channel)
        self.end = end_layer

    def scale(self, x):
        """
        x = [batchsize , n_mel_channels x frames]
        """
        self.mm = torch.mean(x, dim=1, keepdim=True)
        self.std = torch.std(x, dim=1, keepdim=True) + 1e-05
        return (x - self.mm) / self.std

    def forward(self, forward_input):
        """
        forward_input = mel spectrongram of 11 input frames: [batchsize , n_mel_channels , frames]
        """
        forward_input = forward_input.contiguous().view(forward_input.size(
            0), -1)
        output = self.start(self.scale(forward_input))
        output = self.act_func(output)
        for i in range(self.n_layer - 1):
            output = self.fc_layers[i](output)
            output = self.act_func(output)
        output = self.end(output)
        return output


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'n_layer': 1, 'n_in_channel': 4, 'n_out_channel': 4}]
