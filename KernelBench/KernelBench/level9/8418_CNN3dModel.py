import torch


class CNN3dModel(torch.nn.ModuleDict):

    def __init__(self, D_in=1, D_out=1):
        super(CNN3dModel, self).__init__()
        self.conv3d = torch.nn.Conv3d(D_in, D_in * 2, kernel_size=2, stride
            =2, padding=1)
        self.conv3d2 = torch.nn.Conv3d(D_in * 2, D_in * 2, kernel_size=2,
            stride=2, padding=1)
        self.conv3d3 = torch.nn.Conv3d(D_in * 2, D_in * 4, kernel_size=2,
            stride=2, padding=1)
        self.pool = torch.nn.MaxPool3d(kernel_size=2, padding=1)
        self.pool2 = torch.nn.MaxPool3d(kernel_size=2)
        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(D_in * 4, D_in * 8)
        self.linear2 = torch.nn.Linear(D_in * 8, D_out)

    def forward(self, x):
        x = x.float()
        c1 = self.conv3d(x)
        p1 = self.pool(c1)
        c2 = self.conv3d2(self.relu(p1))
        p2 = self.pool(c2)
        c3 = self.conv3d3(self.relu(p2))
        p3 = self.pool2(c3)
        v1 = p3.view(p3.size(0), -1)
        l1 = self.relu(self.linear(v1))
        l2 = self.linear2(l1)
        return l2


def get_inputs():
    return [torch.rand([4, 1, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
