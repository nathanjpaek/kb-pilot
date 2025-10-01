import torch


class MyActivation(torch.nn.Module):

    def __init__(self):
        super(MyActivation, self).__init__()
        self.relu = torch.nn.ReLU6(inplace=False)

    def forward(self, x):
        return x * self.relu(x + 3) / 6


class testHSwish(torch.nn.Module):

    def __init__(self):
        super(testHSwish, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv1 = torch.nn.Conv2d(3, 13, 3)
        self.conv2 = torch.nn.Conv2d(13, 3, 3)
        self.hswish = MyActivation()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.hswish(x)
        return self.dequant(x)


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
