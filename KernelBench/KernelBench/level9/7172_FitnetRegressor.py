import torch
import torch.nn.functional as F


class FitnetRegressor(torch.nn.Module):

    def __init__(self, in_feature, out_feature):
        super(FitnetRegressor, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.regressor = torch.nn.Conv2d(in_feature, out_feature, 1, bias=False
            )
        torch.nn.init.kaiming_normal_(self.regressor.weight, mode='fan_out',
            nonlinearity='relu')
        self.regressor.weight.data.uniform_(-0.005, 0.005)

    def forward(self, feature):
        if feature.dim() == 2:
            feature = feature.unsqueeze(2).unsqueeze(3)
        return F.relu(self.regressor(feature))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_feature': 4, 'out_feature': 4}]
