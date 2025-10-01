import torch
import torch.nn as nn


class WeldonPooling(nn.Module):

    def __init__(self, nMax=1, nMin=None):
        super(WeldonPooling, self).__init__()
        self.nMax = nMax
        if nMin is None:
            self.nMin = nMax
        else:
            self.nMin = nMin
        self.input = torch.Tensor()
        self.output = torch.Tensor()
        self.indicesMax = torch.Tensor()
        self.indicesMin = torch.Tensor()

    def forward(self, input):
        self.batchSize = 0
        self.numChannels = 0
        self.h = 0
        self.w = 0
        if input.dim() == 4:
            self.batchSize = input.size(0)
            self.numChannels = input.size(1)
            self.h = input.size(2)
            self.w = input.size(3)
        elif input.dim() == 3:
            self.batchSize = 1
            self.numChannels = input.size(0)
            self.h = input.size(1)
            self.w = input.size(2)
        else:
            None
        self.input = input
        nMax = self.nMax
        if nMax <= 0:
            nMax = 0
        elif nMax < 1:
            nMax = torch.clamp(torch.floor(nMax * self.h * self.w), min=1)
        nMin = self.nMin
        if nMin <= 0:
            nMin = 0
        elif nMin < 1:
            nMin = torch.clamp(torch.floor(nMin * self.h * self.w), min=1)
        x = input.view(self.batchSize, self.numChannels, self.h * self.w)
        scoreSorted, indices = torch.sort(x, x.dim() - 1, True)
        self.indicesMax = indices[:, :, 0:nMax]
        self.output = torch.sum(scoreSorted[:, :, 0:nMax], dim=2, keepdim=True)
        self.output = self.output.div(nMax)
        if nMin > 0:
            self.indicesMin = indices[:, :, self.h * self.w - nMin:self.h *
                self.w]
            yMin = torch.sum(scoreSorted[:, :, self.h * self.w - nMin:self.
                h * self.w], 2, keepdim=True).div(nMin)
            self.output = torch.add(self.output, yMin)
        if input.dim() == 4:
            self.output = self.output.view(self.batchSize, self.numChannels,
                1, 1)
        elif input.dim() == 3:
            self.output = self.output.view(self.numChannels, 1, 1)
        return self.output

    def backward(self, grad_output, _indices_grad=None):
        nMax = self.nMax
        if nMax <= 0:
            nMax = 0
        elif nMax < 1:
            nMax = torch.clamp(torch.floor(nMax * self.h * self.w), min=1)
        nMin = self.nMin
        if nMin <= 0:
            nMin = 0
        elif nMin < 1:
            nMin = torch.clamp(torch.floor(nMin * self.h * self.w), min=1)
        yMax = grad_output.clone().view(self.batchSize, self.numChannels, 1
            ).expand(self.batchSize, self.numChannels, nMax)
        z = torch.zeros(self.batchSize, self.numChannels, self.h * self.w
            ).type_as(self.input)
        z = z.scatter_(2, self.indicesMax, yMax).div(nMax)
        if nMin > 0:
            yMin = grad_output.clone().view(self.batchSize, self.numChannels, 1
                ).div(nMin).expand(self.batchSize, self.numChannels, nMin)
            self.gradInput = z.scatter_(2, self.indicesMin, yMin).view(self
                .batchSize, self.numChannels, self.h, self.w)
        else:
            self.gradInput = z.view(self.batchSize, self.numChannels, self.
                h, self.w)
        if self.input.dim() == 3:
            self.gradInput = self.gradInput.view(self.numChannels, self.h,
                self.w)
        return self.gradInput


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
