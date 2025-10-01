import torch


class four_layer_conv(torch.nn.Module):

    def __init__(self):
        super(four_layer_conv, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.fcn1 = torch.nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.fcn2 = torch.nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.fcn3 = torch.nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.fcn4 = torch.nn.Conv2d(256, 256, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.relu(self.fcn1(x))
        x = self.relu(self.fcn2(x))
        x = self.relu(self.fcn3(x))
        x = self.relu(self.fcn4(x))
        return x


def get_inputs():
    return [torch.rand([4, 256, 64, 64])]


def get_init_inputs():
    return [[], {}]
