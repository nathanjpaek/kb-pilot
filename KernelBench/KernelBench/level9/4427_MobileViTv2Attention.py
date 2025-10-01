import torch
from torch import nn
from torch.nn import init


class MobileViTv2Attention(nn.Module):
    """
    Scaled dot-product attention
    """

    def __init__(self, d_model):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(MobileViTv2Attention, self).__init__()
        self.fc_i = nn.Linear(d_model, 1)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        self.d_model = d_model
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        """
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :return:
        """
        i = self.fc_i(input)
        weight_i = torch.softmax(i, dim=1)
        context_score = weight_i * self.fc_k(input)
        context_vector = torch.sum(context_score, dim=1, keepdim=True)
        v = self.fc_v(input) * context_vector
        out = self.fc_o(v)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4}]
