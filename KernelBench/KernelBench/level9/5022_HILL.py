import torch
import torch.nn as nn


class HILL(nn.Module):

    def __init__(self, img_size):
        super(HILL, self).__init__()
        self.img_size = img_size
        self.pad_3 = nn.ReplicationPad2d(3)
        self.pad = nn.ReplicationPad2d(7)
        self.conv1 = nn.Conv2d(1, 1, 3, 1, padding=1, bias=False)
        self.avepool1 = nn.AvgPool2d(3, stride=1, padding=1)
        self.avepool2 = nn.AvgPool2d(15, stride=1)
        self.eps = 1e-10
        self.res()

    def res(self):
        self.conv1.weight.data = torch.tensor([[-1, 2, -1], [2, -4, 2], [-1,
            2, -1]], dtype=torch.float).view(1, 1, 3, 3)

    def forward(self, x):
        t1 = self.pad_3(x)
        t2 = self.conv1(t1)
        t3 = self.avepool1(torch.abs(t2))
        t4 = 1 / (t3[:, :, 3:self.img_size + 3, 3:self.img_size + 3] + self.eps
            )
        t5 = self.avepool2(self.pad(t4))
        return t5


def get_inputs():
    return [torch.rand([4, 1, 4, 4])]


def get_init_inputs():
    return [[], {'img_size': 4}]
