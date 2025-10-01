import math
import torch
from torch import nn


class ScaledDotProduct(nn.Module):

    def __init__(self, attentionHeadSize, dropOutProb=0.1):
        super(ScaledDotProduct, self).__init__()
        self.attentionHeadSize = attentionHeadSize
        self.dropout = nn.Dropout(dropOutProb)

    def forward(self, Q, K, V, attentionMask):
        aScores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.
            attentionHeadSize)
        aScores = aScores + attentionMask
        aProbs = self.dropout(nn.Softmax(dim=-1)(aScores))
        return torch.matmul(aProbs, V)


class MultiHeadAttention(nn.Module):

    def __init__(self, hiddenSize, numAttentionHeads, dropoutProb):
        super(MultiHeadAttention, self).__init__()
        if hiddenSize % numAttentionHeads != 0:
            raise ValueError(
                'The hidden size ({}) is not a multiple of the numeber of attention heads ({})'
                .format(hiddenSize, numAttentionHeads))
        self.numAttentionHeads = numAttentionHeads
        self.attentionHeadSize = hiddenSize // self.numAttentionHeads
        self.queriesLinear = nn.Linear(hiddenSize, hiddenSize)
        self.keysLinear = nn.Linear(hiddenSize, hiddenSize)
        self.valuesLinear = nn.Linear(hiddenSize, hiddenSize)
        self.sdp = ScaledDotProduct(self.attentionHeadSize, dropoutProb)
        self.outputLinear = nn.Linear(hiddenSize, hiddenSize)

    def prepareForScores(self, input):
        newShape = input.size()[:-1] + (self.numAttentionHeads, self.
            attentionHeadSize)
        input = input.view(*newShape)
        return input.permute(0, 2, 1, 3)

    def forward(self, hiddenStates, attentionMask):
        projQ = self.queriesLinear(hiddenStates)
        projK = self.keysLinear(hiddenStates)
        projV = self.valuesLinear(hiddenStates)
        queries = self.prepareForScores(projQ)
        keys = self.prepareForScores(projK)
        values = self.prepareForScores(projV)
        attentionScores = self.sdp(queries, keys, values, attentionMask)
        attentionScores = attentionScores.permute(0, 2, 1, 3).contiguous()
        newShape = attentionScores.size()[:-2] + (self.numAttentionHeads *
            self.attentionHeadSize,)
        attentionScores = attentionScores.view(*newShape)
        return self.outputLinear(attentionScores)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hiddenSize': 4, 'numAttentionHeads': 4, 'dropoutProb': 0.5}]
