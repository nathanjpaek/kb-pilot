import torch


class LossCrossentropyAgg(torch.nn.Module):

    def __init__(self):
        super(LossCrossentropyAgg, self).__init__()

    def forward(self, preds, target):
        """ Modified crossentropy that aggregates allowed output classes into single class. """
        preds = torch.clamp(preds, min=1e-10, max=1.0)
        ll_cce_agg = -torch.log(torch.mean(target * preds, dim=1, keepdim=
            False))
        return ll_cce_agg


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
