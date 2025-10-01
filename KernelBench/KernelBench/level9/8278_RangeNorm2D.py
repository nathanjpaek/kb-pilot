import torch
import torch.nn as nn
import torch.autograd


class RangeNorm2D(nn.Module):
    """
        This will normalize a saliency map to range from 0 to 1 via linear range function. 
        
        Input and output will be a 3D tensor of size [batch size x height x width]. 
        
        Input can be any real valued number (supported by hardware)
        Output will range from 0 to 1
        
        Parameters:
            full_norm:     This forces the values to range completely from 0 to 1. 
    """

    def __init__(self, full_norm=True, eps=1e-09):
        super(RangeNorm2D, self).__init__()
        self.full_norm = full_norm
        self.eps = eps

    def forward(self, x):
        """
            Input: 
                x:     A Torch Tensor image with shape [batch size x height x width] e.g. [64,7,7]
                       All values should be real positive (i.e. >= 0).  
            Return:
                x:     x Normalized by dividing by either the min value or the range between max and min. 
                       Each max/min is computed for each batch item.  
        """
        assert torch.is_tensor(x), 'Input must be a Torch Tensor'
        assert len(x.size()
            ) == 3, 'Input should be sizes [batch size x height x width]'
        s0 = x.size()[0]
        s1 = x.size()[1]
        s2 = x.size()[2]
        x = x.reshape(s0, s1 * s2)
        xmax = x.max(dim=1)[0].reshape(s0, 1)
        if self.full_norm:
            xmin = x.min(dim=1)[0].reshape(s0, 1)
            nval = x - xmin
            range = xmax - xmin
        else:
            nval = x
            range = xmax
        """
            prevent divide by zero by setting zero to a small number
            
            Simply adding eps does not work will in this case. So we use torch.where to set a minimum value. 
        """
        eps_mat = torch.zeros_like(range) + self.eps
        range = torch.where(range > self.eps, range, eps_mat)
        x = nval / range
        x = x.reshape(s0, s1, s2)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
