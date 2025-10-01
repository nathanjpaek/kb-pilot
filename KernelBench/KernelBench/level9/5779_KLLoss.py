import torch
from torch import Tensor


class KLLoss(torch.nn.KLDivLoss):

    def __init__(self, batch_wise=False):
        super(KLLoss, self).__init__(reduction='batchmean')
        self.batch_wise = batch_wise

    def forward(self, input: 'Tensor', target: 'Tensor') ->Tensor:
        if self.batch_wise:
            n_labels = target.size()[1]
            target = target.sum(dim=0)
            input = input.argmax(dim=1)
            input = torch.Tensor([input.eq(label).sum() for label in range(
                n_labels)])
        input = torch.nn.LogSigmoid()(input)
        return super().forward(input, target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
