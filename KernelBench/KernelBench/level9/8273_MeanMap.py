import torch
import torch.nn as nn
import torch.autograd


class MeanMap(nn.Module):
    """
        Compute vanilla mean on a 4D tensor. This acts as a standard PyTorch layer. 
        
        The Mean is computed independantly for each batch item at each location x,y
    
        Input should be:
        
        (1) A tensor of size [batch x channels x height x width] 
        (2) Recommend a tensor with only positive values. (After a ReLU)
            Any real value will work. 
        
        Output is a 3D tensor of size [batch x height x width]
    """

    def __init__(self):
        super(MeanMap, self).__init__()

    def forward(self, x):
        assert torch.is_tensor(x), 'input must be a Torch Tensor'
        assert len(x.size()) > 2, 'input must have at least three dims'
        x = torch.mean(x, dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
