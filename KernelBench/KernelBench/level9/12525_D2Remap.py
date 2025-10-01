import torch


class D2Remap(torch.nn.Module):

    def __init__(self):
        super(D2Remap, self).__init__()
        self.l1 = torch.nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.l2 = torch.nn.Conv2d(16, 3, kernel_size=3, padding=1)

    def forward(self, x, depth):
        stack = torch.cat((x, depth.unsqueeze(1)), dim=1)
        return self.l2(self.l1(stack))


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
