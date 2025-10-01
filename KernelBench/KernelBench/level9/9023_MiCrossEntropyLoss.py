import torch


class MiCrossEntropyLoss(torch.nn.Module):

    def __init__(self):
        super(MiCrossEntropyLoss, self).__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, mi_cls_output, label, **_):
        return self.ce_loss(mi_cls_output, label).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
