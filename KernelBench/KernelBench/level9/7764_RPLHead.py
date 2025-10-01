import torch
import torch.nn as nn


class RPLHead(nn.Module):

    def __init__(self, in_channels, num_classes, num_centers=1, init='random'):
        super(RPLHead, self).__init__()
        self.feat_dim = in_channels
        self.num_classes = num_classes
        self.num_centers = num_centers
        if init == 'random':
            self.centers = nn.Parameter(0.1 * torch.randn(num_classes *
                num_centers, self.feat_dim))
        else:
            self.centers = nn.Parameter(torch.Tensor(num_classes *
                num_centers, self.feat_dim))
            self.centers.data.fill_(0)

    def forward(self, raw_features, center=None, metric='l2'):
        """ features: (B, D, T)
        """
        num_times = raw_features.size(-1)
        features = raw_features.permute(0, 2, 1).contiguous().view(-1, self
            .feat_dim)
        if metric == 'l2':
            f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
            if center is None:
                c_2 = torch.sum(torch.pow(self.centers, 2), dim=1, keepdim=True
                    )
                dist = f_2 - 2 * torch.matmul(features, torch.transpose(
                    self.centers, 1, 0)) + torch.transpose(c_2, 1, 0)
            else:
                c_2 = torch.sum(torch.pow(center, 2), dim=1, keepdim=True)
                dist = f_2 - 2 * torch.matmul(features, torch.transpose(
                    center, 1, 0)) + torch.transpose(c_2, 1, 0)
            dist = dist / float(features.shape[1])
        else:
            if center is None:
                center = self.centers
            else:
                center = center
            dist = features.matmul(center.t())
        dist = torch.reshape(dist, [-1, self.num_classes, self.num_centers])
        dist = torch.mean(dist, dim=2)
        dist = dist.view(-1, num_times, self.num_classes).permute(0, 2, 1
            ).contiguous()
        return dist


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'num_classes': 4}]
