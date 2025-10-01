import torch
import torch.nn as nn
from torch.nn import Linear


class fusion(nn.Module):

    def __init__(self, feature_size=768):
        super(fusion, self).__init__()
        self.fc1 = Linear(feature_size * 3, 1)
        self.fc2 = Linear(feature_size * 3, 1)
        self.fc3 = Linear(feature_size * 3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x3):
        batch_size = x1.size()[0]
        x1 = x1.view(-1, 768)
        x2 = x2.view(-1, 768)
        x3 = x3.view(-1, 768)
        x123 = torch.cat((x1, x2), 1)
        x123 = torch.cat((x123, x3), 1)
        weight1 = self.fc1(x123)
        weight2 = self.fc2(x123)
        weight3 = self.fc3(x123)
        weight1 = self.sigmoid(weight1)
        weight2 = self.sigmoid(weight2)
        weight3 = self.sigmoid(weight3)
        weight1 = weight1.view(batch_size, -1).unsqueeze(2)
        weight2 = weight1.view(batch_size, -1).unsqueeze(2)
        weight3 = weight1.view(batch_size, -1).unsqueeze(2)
        return weight1, weight2, weight3


def get_inputs():
    return [torch.rand([4, 768]), torch.rand([4, 768]), torch.rand([4, 768])]


def get_init_inputs():
    return [[], {}]
