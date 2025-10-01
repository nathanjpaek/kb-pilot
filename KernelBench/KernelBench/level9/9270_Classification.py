import torch
import torch.nn as nn


class Classification(nn.Module):
    """一个最简单的一层分类模型
    Parameters:
        input_size:输入维度
        num_classes:类别数量
    return:
        logists:最大概率对应的标签
    """

    def __init__(self, input_size, num_classes):
        super(Classification, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        logists = torch.log_softmax(self.fc1(x), 1)
        return logists


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'num_classes': 4}]
