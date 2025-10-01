import math
import torch
from torch import nn


class NormLayer(nn.Module):
    """
		Implementation of Layer Normalization (https://arxiv.org/abs/1607.06450)
		It consists of Batch Normalization Transform to speed up learning with mean and std computed according to the above paper

		normWeights:
			weights for this normalization layer which will be learnt during training
		normBias:
			bias for this normalization layer which will be learnt during training
		epsilon:
			numerical stability parameter to avoid division by zero
	"""

    def __init__(self, hiddenSize, epsilon=1e-12):
        super(NormLayer, self).__init__()
        self.weight = nn.Parameter(torch.ones(hiddenSize))
        self.bias = nn.Parameter(torch.zeros(hiddenSize))
        self.epsilon = epsilon

    def forward(self, input):
        mu = input.mean(-1, keepdim=True)
        stdArg = (input - mu).pow(2).mean(-1, keepdim=True) + self.epsilon
        std = torch.sqrt(stdArg)
        input = (input - mu) / std
        return self.weight * input + self.bias


class GELU(nn.Module):

    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, tensor):
        geluPow = tensor + 0.044715 * torch.pow(tensor, 3)
        geluTanh = torch.tanh(math.sqrt(2 / math.pi) * geluPow)
        geluResult = 1 + geluTanh
        return 0.5 * tensor * geluResult


class FeedForward(nn.Module):

    def __init__(self, hiddenSize, innerLayerDimension, dropOutProb=0.1):
        super(FeedForward, self).__init__()
        self.activationFuncion = GELU()
        self.dropout = nn.Dropout(dropOutProb)
        self.w1 = nn.Linear(hiddenSize, innerLayerDimension)
        self.w2 = nn.Linear(innerLayerDimension, hiddenSize)

    def forward(self, tensor):
        intermediate = self.activationFuncion(self.w1(tensor))
        linearOut = self.w2(intermediate)
        return self.dropout(linearOut)


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


class Encoder(nn.Module):

    def __init__(self, hiddenSize, numAttentionHeads, normEpsilon=1e-12,
        dropoutProb=0.1):
        super(Encoder, self).__init__()
        self.multiHeadAtt = MultiHeadAttention(hiddenSize,
            numAttentionHeads, dropoutProb)
        self.feedForward = FeedForward(hiddenSize, 4 * hiddenSize, dropoutProb)
        self.attNorm = NormLayer(hiddenSize, normEpsilon)
        self.outputNorm = NormLayer(hiddenSize, normEpsilon)
        self.dropout = nn.Dropout(dropoutProb)

    def forward(self, hiddenStates, attentionMask):
        attentionOutput = self.multiHeadAtt(hiddenStates, attentionMask)
        attentionOutput = self.dropout(attentionOutput)
        normAttOutput = self.attNorm(attentionOutput + hiddenStates)
        ffOutput = self.feedForward(normAttOutput)
        normFFOutput = self.outputNorm(ffOutput + normAttOutput)
        return normFFOutput


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hiddenSize': 4, 'numAttentionHeads': 4}]
