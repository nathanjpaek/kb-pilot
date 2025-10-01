import torch
import torch.nn.functional as F


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=6, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=6, stride=1, padding=1
            )
        self.conv3 = torch.nn.Conv2d(32, 10, kernel_size=6, stride=1, padding=1
            )

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.adaptive_avg_pool2d(x, 1)
        return x.view(batch_size, 10)


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
