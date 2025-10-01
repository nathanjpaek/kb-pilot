import torch
import torch.nn as nn


class SSRLayer(nn.Module):

    def __init__(self):
        super(SSRLayer, self).__init__()

    def forward(self, x):
        a = x[0][:, :, 0] * 0
        b = x[0][:, :, 0] * 0
        c = x[0][:, :, 0] * 0
        s1 = 3
        s2 = 3
        s3 = 3
        lambda_d = 1
        di = s1 // 2
        dj = s2 // 2
        dk = s3 // 2
        V = 99
        for i in range(0, s1):
            a = a + (i - di + x[6]) * x[0][:, :, i]
        a = a / (s1 * (1 + lambda_d * x[3]))
        for j in range(0, s2):
            b = b + (j - dj + x[7]) * x[1][:, :, j]
        b = b / (s1 * (1 + lambda_d * x[3])) / (s2 * (1 + lambda_d * x[4]))
        for k in range(0, s3):
            c = c + (k - dk + x[8]) * x[2][:, :, k]
        c = c / (s1 * (1 + lambda_d * x[3])) / (s2 * (1 + lambda_d * x[4])) / (
            s3 * (1 + lambda_d * x[5]))
        pred = (a + b + c) * V
        return pred


def get_inputs():
    return [torch.rand([9, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
