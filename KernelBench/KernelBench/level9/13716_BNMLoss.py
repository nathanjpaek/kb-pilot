import torch


class BNMLoss(torch.nn.Module):
    """
    Implementation of the loss in
    [Towards Discriminability and Diversity:
    Batch Nuclear-norm Maximization
    under Label Insufficient Situations](https://arxiv.org/abs/2003.12237).
    """

    def forward(self, x):
        """"""
        x = torch.nn.functional.softmax(x, dim=1)
        return -torch.linalg.norm(x, 'nuc') / x.shape[0]


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
