import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data.distributed


class SpatialAttentionLayer(nn.Module):

    def __init__(self, spatial_size):
        super(SpatialAttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.width_w0 = nn.Parameter(torch.ones(spatial_size, 1),
            requires_grad=True)
        self.width_w1 = nn.Parameter(torch.ones(spatial_size, 1),
            requires_grad=True)
        self.width_w2 = nn.Parameter(torch.ones(spatial_size, 1),
            requires_grad=True)
        self.width_bias0 = nn.Parameter(torch.zeros(1, spatial_size),
            requires_grad=True)
        self.width_bias1 = nn.Parameter(torch.zeros(1, spatial_size),
            requires_grad=True)
        self.width_bias2 = nn.Parameter(torch.zeros(1, spatial_size),
            requires_grad=True)
        self.height_w0 = nn.Parameter(torch.ones(spatial_size, 1),
            requires_grad=True)
        self.height_w1 = nn.Parameter(torch.ones(spatial_size, 1),
            requires_grad=True)
        self.height_w2 = nn.Parameter(torch.ones(spatial_size, 1),
            requires_grad=True)
        self.height_bias0 = nn.Parameter(torch.zeros(1, spatial_size),
            requires_grad=True)
        self.height_bias1 = nn.Parameter(torch.zeros(1, spatial_size),
            requires_grad=True)
        self.height_bias2 = nn.Parameter(torch.zeros(1, spatial_size),
            requires_grad=True)
        nn.init.xavier_uniform_(self.width_w0)
        nn.init.xavier_uniform_(self.width_w1)
        nn.init.xavier_uniform_(self.width_w2)
        nn.init.xavier_uniform_(self.height_w0)
        nn.init.xavier_uniform_(self.height_w1)
        nn.init.xavier_uniform_(self.height_w2)

    def forward(self, x):
        b, _c, h, w = x.size()
        x_spatial_max = torch.max(x, 1)[0]
        x_spatial_mean = torch.mean(x, 1)
        x_width_max = torch.max(x_spatial_max, 1)[0]
        x_width_mean = torch.mean(x_spatial_mean, 1)
        x_height_max = torch.max(x_spatial_max, 2)[0]
        x_height_mean = torch.mean(x_spatial_mean, 2)
        x0_w_s = self.softmax(x_width_mean)
        y0_w = torch.matmul(x_width_mean, self.width_w0)
        y1_w = torch.matmul(x_width_max, self.width_w1)
        y0_w_t = torch.tanh(y0_w * x0_w_s + self.width_bias0)
        y1_w_t = torch.tanh(y1_w * x0_w_s + self.width_bias1)
        y2_w = torch.matmul(y1_w_t, self.width_w2)
        y2_w_t = y2_w * y0_w_t + self.width_bias2
        x0_h_s = self.softmax(x_height_mean)
        y0_h = torch.matmul(x_height_mean, self.height_w0)
        y1_h = torch.matmul(x_height_max, self.height_w1)
        y0_h_t = torch.tanh(y0_h * x0_h_s + self.height_bias0)
        y1_h_t = torch.tanh(y1_h * x0_h_s + self.height_bias1)
        y2_h = torch.matmul(y1_h_t, self.height_w2)
        y2_h_t = y2_h * y0_h_t + self.height_bias2
        spatial = torch.tanh(torch.matmul(y2_h_t.view(b, h, 1), y2_w_t.view
            (b, 1, w))).unsqueeze(1)
        z = x * (spatial + 1)
        return z


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'spatial_size': 4}]
