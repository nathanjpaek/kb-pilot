import torch
import torch.nn as nn


class FCNet(nn.Module):

    def __init__(self, in_size, out_size, activate=None, drop=0.0):
        super(FCNet, self).__init__()
        self.lin = nn.Linear(in_size, out_size)
        self.drop_value = drop
        self.drop = nn.Dropout(drop)
        self.activate = activate.lower() if activate is not None else None
        if activate == 'relu':
            self.ac_fn = nn.ReLU()
        elif activate == 'sigmoid':
            self.ac_fn = nn.Sigmoid()
        elif activate == 'tanh':
            self.ac_fn = nn.Tanh()

    def forward(self, x):
        if self.drop_value > 0:
            x = self.drop(x)
        x = self.lin(x)
        if self.activate is not None:
            x = self.ac_fn(x)
        return x


class Classifier(nn.Sequential):

    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.lin1 = FCNet(in_features, mid_features, activate='relu', drop=drop
            )
        self.lin2 = FCNet(mid_features, out_features, drop=drop)
        self.bilinear = nn.Bilinear(in1_features=in_features, in2_features=
            in_features, out_features=mid_features)

    def forward(self, v, q):
        """
        :param v: [batch, r1, features]
        :param q: [batch, r2, features]
        :return:
        """
        num_obj = v.shape[2]
        max_len = q.shape[2]
        v_mean = v.sum(1) / num_obj
        q_mean = q.sum(1) / max_len
        out = self.lin1(v_mean * q_mean)
        out = self.bilinear(v_mean, q_mean)
        out = self.lin2(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'mid_features': 4, 'out_features': 4}]
