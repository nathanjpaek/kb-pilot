import torch
import torch.nn as nn
import torch.onnx
from itertools import product as product


def gen_nonlinearity(A, nonlinearity):
    """
    Returns required activation for a tensor based on the inputs

    nonlinearity is either a callable or a value in
        ['tanh', 'sigmoid', 'relu', 'quantTanh', 'quantSigm', 'quantSigm4']
    """
    if nonlinearity == 'tanh':
        return torch.tanh(A)
    elif nonlinearity == 'sigmoid':
        return torch.sigmoid(A)
    elif nonlinearity == 'relu':
        return torch.relu(A, 0.0)
    elif nonlinearity == 'quantTanh':
        return torch.max(torch.min(A, torch.ones_like(A)), -1.0 * torch.
            ones_like(A))
    elif nonlinearity == 'quantSigm':
        A = (A + 1.0) / 2.0
        return torch.max(torch.min(A, torch.ones_like(A)), torch.zeros_like(A))
    elif nonlinearity == 'quantSigm4':
        A = (A + 2.0) / 4.0
        return torch.max(torch.min(A, torch.ones_like(A)), torch.zeros_like(A))
    else:
        if not callable(nonlinearity):
            raise ValueError(
                'nonlinearity is either a callable or a value ' +
                "['tanh', 'sigmoid', 'relu', 'quantTanh', " + "'quantSigm'")
        return nonlinearity(A)


class RNNCell(nn.Module):

    def __init__(self, input_size, hidden_size, gate_nonlinearity,
        update_nonlinearity, num_W_matrices, num_U_matrices, num_biases,
        wRank=None, uRank=None, wSparsity=1.0, uSparsity=1.0):
        super(RNNCell, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._gate_nonlinearity = gate_nonlinearity
        self._update_nonlinearity = update_nonlinearity
        self._num_W_matrices = num_W_matrices
        self._num_U_matrices = num_U_matrices
        self._num_biases = num_biases
        self._num_weight_matrices = [self._num_W_matrices, self.
            _num_U_matrices, self._num_biases]
        self._wRank = wRank
        self._uRank = uRank
        self._wSparsity = wSparsity
        self._uSparsity = uSparsity
        self.oldmats = []

    @property
    def state_size(self):
        return self._hidden_size

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def gate_nonlinearity(self):
        return self._gate_nonlinearity

    @property
    def update_nonlinearity(self):
        return self._update_nonlinearity

    @property
    def wRank(self):
        return self._wRank

    @property
    def uRank(self):
        return self._uRank

    @property
    def num_W_matrices(self):
        return self._num_W_matrices

    @property
    def num_U_matrices(self):
        return self._num_U_matrices

    @property
    def num_weight_matrices(self):
        return self._num_weight_matrices

    @property
    def name(self):
        raise NotImplementedError()

    def forward(self, input, state):
        raise NotImplementedError()

    def getVars(self):
        raise NotImplementedError()

    def get_model_size(self):
        """
        Function to get aimed model size
        """
        mats = self.getVars()
        endW = self._num_W_matrices
        endU = endW + self._num_U_matrices
        totalnnz = 2
        for i in range(0, endW):
            mats[i].device
            totalnnz += utils.countNNZ(mats[i].cpu(), self._wSparsity)
            mats[i]
        for i in range(endW, endU):
            mats[i].device
            totalnnz += utils.countNNZ(mats[i].cpu(), self._uSparsity)
            mats[i]
        for i in range(endU, len(mats)):
            mats[i].device
            totalnnz += utils.countNNZ(mats[i].cpu(), False)
            mats[i]
        return totalnnz * 4

    def copy_previous_UW(self):
        mats = self.getVars()
        num_mats = self._num_W_matrices + self._num_U_matrices
        if len(self.oldmats) != num_mats:
            for i in range(num_mats):
                self.oldmats.append(torch.FloatTensor())
        for i in range(num_mats):
            self.oldmats[i] = torch.FloatTensor(mats[i].detach().clone())

    def sparsify(self):
        mats = self.getVars()
        endW = self._num_W_matrices
        endU = endW + self._num_U_matrices
        for i in range(0, endW):
            mats[i] = utils.hardThreshold(mats[i], self._wSparsity)
        for i in range(endW, endU):
            mats[i] = utils.hardThreshold(mats[i], self._uSparsity)
        self.copy_previous_UW()

    def sparsifyWithSupport(self):
        mats = self.getVars()
        endU = self._num_W_matrices + self._num_U_matrices
        for i in range(0, endU):
            mats[i] = utils.supportBasedThreshold(mats[i], self.oldmats[i])


class FastRNNCell(RNNCell):
    """
    FastRNN Cell with Both Full Rank and Low Rank Formulations
    Has multiple activation functions for the gates
    hidden_size = # hidden units

    update_nonlinearity = nonlinearity for final rnn update
    can be chosen from [tanh, sigmoid, relu, quantTanh, quantSigm]

    wRank = rank of W matrix (creates two matrices if not None)
    uRank = rank of U matrix (creates two matrices if not None)

    wSparsity = intended sparsity of W matrix(ces)
    uSparsity = intended sparsity of U matrix(ces)
    Warning:
    The Cell will not automatically sparsify.
    The user must invoke .sparsify to hard threshold.

    alphaInit = init for alpha, the update scalar
    betaInit = init for beta, the weight for previous state

    FastRNN architecture and compression techniques are found in
    FastGRNN(LINK) paper

    Basic architecture is like:

    h_t^ = update_nl(Wx_t + Uh_{t-1} + B_h)
    h_t = sigmoid(beta)*h_{t-1} + sigmoid(alpha)*h_t^

    W and U can further parameterised into low rank version by
    W = matmul(W_1, W_2) and U = matmul(U_1, U_2)
    """

    def __init__(self, input_size, hidden_size, update_nonlinearity='tanh',
        wRank=None, uRank=None, wSparsity=1.0, uSparsity=1.0, alphaInit=-
        3.0, betaInit=3.0, name='FastRNN'):
        super(FastRNNCell, self).__init__(input_size, hidden_size, None,
            update_nonlinearity, 1, 1, 1, wRank, uRank, wSparsity, uSparsity)
        self._alphaInit = alphaInit
        self._betaInit = betaInit
        if wRank is not None:
            self._num_W_matrices += 1
            self._num_weight_matrices[0] = self._num_W_matrices
        if uRank is not None:
            self._num_U_matrices += 1
            self._num_weight_matrices[1] = self._num_U_matrices
        self._name = name
        if wRank is None:
            self.W = nn.Parameter(0.1 * torch.randn([input_size, hidden_size]))
        else:
            self.W1 = nn.Parameter(0.1 * torch.randn([input_size, wRank]))
            self.W2 = nn.Parameter(0.1 * torch.randn([wRank, hidden_size]))
        if uRank is None:
            self.U = nn.Parameter(0.1 * torch.randn([hidden_size, hidden_size])
                )
        else:
            self.U1 = nn.Parameter(0.1 * torch.randn([hidden_size, uRank]))
            self.U2 = nn.Parameter(0.1 * torch.randn([uRank, hidden_size]))
        self.bias_update = nn.Parameter(torch.ones([1, hidden_size]))
        self.alpha = nn.Parameter(self._alphaInit * torch.ones([1, 1]))
        self.beta = nn.Parameter(self._betaInit * torch.ones([1, 1]))

    @property
    def name(self):
        return self._name

    @property
    def cellType(self):
        return 'FastRNN'

    def forward(self, input, state):
        if self._wRank is None:
            wComp = torch.matmul(input, self.W)
        else:
            wComp = torch.matmul(torch.matmul(input, self.W1), self.W2)
        if self._uRank is None:
            uComp = torch.matmul(state, self.U)
        else:
            uComp = torch.matmul(torch.matmul(state, self.U1), self.U2)
        pre_comp = wComp + uComp
        c = gen_nonlinearity(pre_comp + self.bias_update, self.
            _update_nonlinearity)
        new_h = torch.sigmoid(self.beta) * state + torch.sigmoid(self.alpha
            ) * c
        return new_h

    def getVars(self):
        Vars = []
        if self._num_W_matrices == 1:
            Vars.append(self.W)
        else:
            Vars.extend([self.W1, self.W2])
        if self._num_U_matrices == 1:
            Vars.append(self.U)
        else:
            Vars.extend([self.U1, self.U2])
        Vars.extend([self.bias_update])
        Vars.extend([self.alpha, self.beta])
        return Vars


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
