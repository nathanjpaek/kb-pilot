import torch
from torch import nn


class OHEM_CrossEntroy_Loss(nn.Module):

    def __init__(self, threshold, keep_num):
        super(OHEM_CrossEntroy_Loss, self).__init__()
        self.threshold = threshold
        self.keep_num = keep_num
        self.loss_function = nn.CrossEntropyLoss(reduction='none')

    def forward(self, output, target):
        loss = self.loss_function(output, target).view(-1)
        loss, _loss_index = torch.sort(loss, descending=True)
        threshold_in_keep_num = loss[self.keep_num]
        if threshold_in_keep_num > self.threshold:
            loss = loss[loss > self.threshold]
        else:
            loss = loss[:self.keep_num]
        return torch.mean(loss)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'threshold': 4, 'keep_num': 4}]
