import torch
import torch.nn as torch_nn
from torch.nn import Parameter
import torch.utils


class P2SActivationLayer(torch_nn.Module):
    """ Output layer that produces cos	heta between activation vector x
    and class vector w_j
    in_dim:     dimension of input feature vectors
    output_dim: dimension of output feature vectors 
                (i.e., number of classes)
    
    
    Usage example:
      batchsize = 64
      input_dim = 10
      class_num = 5
      l_layer = P2SActivationLayer(input_dim, class_num)
      l_loss = P2SGradLoss()
      data = torch.rand(batchsize, input_dim, requires_grad=True)
      target = (torch.rand(batchsize) * class_num).clamp(0, class_num-1)
      target = target.to(torch.long)
      scores = l_layer(data)
      loss = l_loss(scores, target)
      loss.backward()
    """

    def __init__(self, in_dim, out_dim):
        super(P2SActivationLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = Parameter(torch.Tensor(in_dim, out_dim))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-05).mul_(100000.0)
        return

    def forward(self, input_feat):
        """
        Compute P2sgrad activation
        
        input:
        ------
          input_feat: tensor (batchsize, input_dim)
        output:
        -------
          tensor (batchsize, output_dim)
          
        """
        w = self.weight.renorm(2, 1, 1e-05).mul(100000.0)
        x_modulus = input_feat.pow(2).sum(1).pow(0.5)
        w.pow(2).sum(0).pow(0.5)
        inner_wx = input_feat.mm(w)
        cos_theta = inner_wx / x_modulus.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)
        return cos_theta


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
