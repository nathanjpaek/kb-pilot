import torch
import torch.nn as nn


class MumfordShah_Loss(nn.Module):

    def levelsetLoss(self, output, target, penalty='l1'):
        outshape = output.shape
        tarshape = target.shape
        self.penalty = penalty
        loss = 0.0
        for ich in range(tarshape[1]):
            target_ = torch.unsqueeze(target[:, ich], 1)
            target_ = target_.expand(tarshape[0], outshape[1], tarshape[2],
                tarshape[3])
            pcentroid = torch.sum(target_ * output, (2, 3)) / torch.sum(output,
                (2, 3))
            pcentroid = pcentroid.view(tarshape[0], outshape[1], 1, 1)
            plevel = target_ - pcentroid.expand(tarshape[0], outshape[1],
                tarshape[2], tarshape[3])
            pLoss = plevel * plevel * output
            loss += torch.sum(pLoss)
        return loss

    def gradientLoss2d(self, input):
        dH = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
        dW = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])
        if self.penalty == 'l2':
            dH = dH * dH
            dW = dW * dW
        loss = torch.sum(dH) + torch.sum(dW)
        return loss

    def forward(self, image, prediction):
        loss_level = self.levelsetLoss(image, prediction)
        loss_tv = self.gradientLoss2d(image)
        return loss_level + loss_tv


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
