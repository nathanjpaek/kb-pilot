import torch


class VonmisesLossBiternion(torch.nn.Module):
    """Von mises loss function for biternion inputs

    see: Beyer et al.: Biternion Nets: Continuous Head Pose Regression from
         Discrete Training Labels, GCPR 2015.
    """

    def __init__(self, kappa):
        super(VonmisesLossBiternion, self).__init__()
        self._kappa = kappa

    def forward(self, prediction, target):
        cos_angles = torch.bmm(prediction[..., None].permute(0, 2, 1),
            target[..., None])
        cos_angles = torch.exp(self._kappa * (cos_angles - 1))
        score = 1 - cos_angles
        return score[:, 0, 0]


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'kappa': 4}]
