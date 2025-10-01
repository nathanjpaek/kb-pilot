import torch
import torch.utils.data
from torch import nn
import torch


class netMSELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        return self.computeLoss(output, target)

    def computeLoss(self, output, target):
        loss = torch.mean((output - target) ** 2)
        return loss


class SuperLoss(nn.Module):

    def __init__(self, Losses=[], Weights=[], Names=[]):
        super().__init__()
        if not Losses:
            self.Losses = [netMSELoss()]
            self.Weights = [1.0]
            self.Names = ['Default MSE Loss']
        else:
            if len(Losses) != len(Weights):
                raise RuntimeError(
                    'SuperLoss() given Losses and Weights dont match.')
            self.Losses = Losses
            self.Weights = Weights
            self.Names = [('Subloss ' + str(i).zfill(2)) for i in range(len
                (self.Losses))]
            for Ctr, n in enumerate(Names, 0):
                self.Names[Ctr] = n
            self.cleanUp()

    def __len__(self):
        return len(self.Losses)

    def getItems(self, withoutWeights=False):
        RetLossValsFloat = []
        if withoutWeights:
            for v in self.LossVals:
                RetLossValsFloat.append(v.item())
        else:
            for v in self.LossValsWeighted:
                RetLossValsFloat.append(v.item())
        return RetLossValsFloat

    def cleanUp(self):
        self.LossVals = [0.0] * len(self.Losses)
        self.LossValsWeighted = [0.0] * len(self.Losses)

    def forward(self, output, target):
        self.cleanUp()
        return self.computeLoss(output, target)

    def computeLoss(self, output, target):
        TotalLossVal = 0.0
        for Ctr, (l, w) in enumerate(zip(self.Losses, self.Weights), 0):
            LossVal = l.forward(output, target)
            self.LossVals[Ctr] = LossVal
            self.LossValsWeighted[Ctr] = w * LossVal
            TotalLossVal += self.LossValsWeighted[Ctr]
        return TotalLossVal


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
