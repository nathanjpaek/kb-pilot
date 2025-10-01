import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx


class BinaryTreeLeafModule(nn.Module):
    """
  local input = nn.Identity()()
  local c = nn.Linear(self.in_dim, self.mem_dim)(input)
  local h
  if self.gate_output then
    local o = nn.Sigmoid()(nn.Linear(self.in_dim, self.mem_dim)(input))
    h = nn.CMulTable(){o, nn.Tanh()(c)}
  else
    h = nn.Tanh()(c)
  end

  local leaf_module = nn.gModule({input}, {c, h})
    """

    def __init__(self, cuda, in_dim, mem_dim):
        super(BinaryTreeLeafModule, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.cx = nn.Linear(self.in_dim, self.mem_dim)
        self.ox = nn.Linear(self.in_dim, self.mem_dim)
        if self.cudaFlag:
            self.cx = self.cx
            self.ox = self.ox

    def forward(self, input):
        c = self.cx(input)
        o = F.sigmoid(self.ox(input))
        h = o * F.tanh(c)
        return c, h


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'cuda': False, 'in_dim': 4, 'mem_dim': 4}]
