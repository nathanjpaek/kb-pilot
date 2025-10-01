import torch


class GreedyHashLoss(torch.nn.Module):

    def __init__(self):
        super(GreedyHashLoss, self).__init__()

    def forward(self, u):
        b = GreedyHashLoss.Hash.apply(u)
        loss = (u.abs() - 1).pow(3).abs().mean()
        return b, loss


    class Hash(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            return input.sign()

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
