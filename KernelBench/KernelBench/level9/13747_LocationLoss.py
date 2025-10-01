import torch


class LocationLoss(torch.nn.Module):

    def __init__(self, crop_size=192, **kwargs):
        super().__init__()
        self._crop_size = crop_size

    def forward(self, pred_locations, teac_locations):
        pred_locations = pred_locations / (0.5 * self._crop_size) - 1
        return torch.mean(torch.abs(pred_locations - teac_locations), dim=(
            1, 2, 3))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
