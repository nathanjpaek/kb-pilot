import torch


class Log_Cosh_Loss(torch.nn.Module):

    def forward(self, logits, labels):
        return torch.mean(torch.log(torch.cosh(labels - logits)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
